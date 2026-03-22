import logging
import os
import subprocess
import threading
import urllib.request
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import ConfigDict, Field
from typing_extensions import Literal

from frigate.const import MODEL_CACHE_DIR
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from frigate.object_detection.util import RequestStore, ResponseStore

logger = logging.getLogger(__name__)


# ----------------- Utility Functions ----------------- #


def preprocess_tensor(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Resize an image with unchanged aspect ratio using padding.
    Assumes input image shape is (H, W, 3).
    """
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    h, w = image.shape[:2]
    scale = min(model_w / w, model_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    padded_image = np.full((model_h, model_w, 3), 114, dtype=image.dtype)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2
    padded_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
        resized_image
    )
    return padded_image


# ----------------- Global Constants ----------------- #
DETECTOR_KEY = "hailo10h"
HAILO10H_ARCH = "hailo10h"
HAILO10H_ARCH_CAPS = "HAILO10H"
SHARED_VDEVICE_GROUP_ID = "SHARED"
H10H_DEFAULT_MODEL = "yolov6n.hef"
H10H_DEFAULT_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov6n.hef"
SUPPORTED_DEVICE_ARCHITECTURES = {HAILO10H_ARCH_CAPS}
DEFAULT_INFERENCE_TIMEOUT = 1.0


@dataclass(frozen=True)
class HailoDeviceInfo:
    architecture: Optional[str]
    firmware_version: Optional[str]
    raw_output: str



def detect_hailo_device_info() -> HailoDeviceInfo:
    """Probe the installed Hailo device via hailortcli.

    On Raspberry Pi 5 / Trixie with an AI HAT+ 2 we expect the reported
    architecture to be HAILO10H. The helper is intentionally isolated so it can
    be unit-tested and reused by future integration checks.
    """
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "hailortcli was not found. Install HailoRT before using the hailo10h detector."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to probe Hailo device information: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown hailortcli error"
        raise RuntimeError(f"hailortcli fw-control identify failed: {stderr}")

    architecture = None
    firmware_version = None

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Device Architecture:"):
            architecture = line.split(":", 1)[1].strip().upper()
        elif line.startswith("Firmware Version:"):
            firmware_version = line.split(":", 1)[1].strip()

    return HailoDeviceInfo(
        architecture=architecture,
        firmware_version=firmware_version,
        raw_output=result.stdout,
    )



def validate_hailo10h_device(device_info: HailoDeviceInfo) -> None:
    if device_info.architecture not in SUPPORTED_DEVICE_ARCHITECTURES:
        found = device_info.architecture or "unknown"
        raise RuntimeError(
            "The hailo10h detector requires a Hailo-10H device, but "
            f"hailortcli reported '{found}'. Expected {HAILO10H_ARCH_CAPS}."
        )



def get_model_hw_from_input_shape(input_shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(input_shape) != 3:
        raise ValueError(
            f"Unsupported Hailo input shape {input_shape!r}. Expected a 3D tensor."
        )

    height, width, channels = input_shape
    if channels != 3:
        raise ValueError(
            f"Unsupported Hailo input shape {input_shape!r}. Expected 3 channels."
        )

    return height, width


# ----------------- HailoAsyncInference Class ----------------- #
class HailoAsyncInference:
    def __init__(
        self,
        hef_path: str,
        input_store: RequestStore,
        output_store: ResponseStore,
        batch_size: int = 1,
        input_type: Optional[str] = None,
        output_type: Optional[Dict[str, str]] = None,
        send_original_frame: bool = False,
    ) -> None:
        # when importing hailo it activates the driver
        # which leaves processes running even though it may not be used.
        try:
            from hailo_platform import (
                HEF,
                FormatType,
                HailoSchedulingAlgorithm,
                VDevice,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "hailo_platform is not installed. Install the HailoRT Python wheel "
                "before using the hailo10h detector."
            ) from exc

        self.input_store = input_store
        self.output_store = output_store

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = SHARED_VDEVICE_GROUP_ID

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        if input_type is not None:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))

        if output_type is not None:
            for output_name, output_type in output_type.items():
                self.infer_model.output(output_name).set_format_type(
                    getattr(FormatType, output_type)
                )

        self.output_type = output_type
        self.send_original_frame = send_original_frame

    def callback(
        self,
        completion_info,
        bindings_list: List,
        input_batch: List,
        request_ids: List[int],
    ):
        if completion_info.exception:
            logger.error(f"Inference error: {completion_info.exception}")
        else:
            for i, bindings in enumerate(bindings_list):
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                        for name in bindings._output_names
                    }
                self.output_store.put(request_ids[i], (input_batch[i], result))

    def _create_bindings(self, configured_infer_model) -> object:
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=getattr(
                        np, str(output_info.format.type).split(".")[1].lower()
                    ),
                )
                for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=getattr(np, self.output_type[name].lower()),
                )
                for name in self.output_type
            }
        return configured_infer_model.create_bindings(output_buffers=output_buffers)

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self) -> None:
        job = None
        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_data = self.input_store.get()

                if batch_data is None:
                    break

                request_id, frame_data = batch_data
                preprocessed_batch = [frame_data]
                request_ids = [request_id]
                input_batch = preprocessed_batch  # non-send_original_frame mode

                bindings_list = []
                for frame in preprocessed_batch:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)
                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                job = configured_infer_model.run_async(
                    bindings_list,
                    partial(
                        self.callback,
                        input_batch=input_batch,
                        request_ids=request_ids,
                        bindings_list=bindings_list,
                    ),
                )

            if job is not None:
                job.wait(100)


# ----------------- HailoDetector Class ----------------- #
class HailoDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: "HailoDetectorConfig"):
        self.cache_dir = MODEL_CACHE_DIR
        self.device_type = detector_config.device
        self.model_height = (
            detector_config.model.height
            if hasattr(detector_config.model, "height")
            else None
        )
        self.model_width = (
            detector_config.model.width
            if hasattr(detector_config.model, "width")
            else None
        )
        self.model_type = (
            detector_config.model.model_type
            if hasattr(detector_config.model, "model_type")
            else None
        )
        self.tensor_format = (
            detector_config.model.input_tensor
            if hasattr(detector_config.model, "input_tensor")
            else None
        )
        self.pixel_format = (
            detector_config.model.input_pixel_format
            if hasattr(detector_config.model, "input_pixel_format")
            else None
        )
        self.input_dtype = (
            detector_config.model.input_dtype
            if hasattr(detector_config.model, "input_dtype")
            else None
        )
        self.output_type = "FLOAT32"
        self.set_path_and_url(detector_config.model.path)
        self.device_info = detect_hailo_device_info()
        validate_hailo10h_device(self.device_info)
        self.working_model_path = self.check_and_prepare()

        self.batch_size = 1
        self.input_store = RequestStore()
        self.response_store = ResponseStore()

        try:
            logger.debug(f"[INIT] Loading HEF model from {self.working_model_path}")
            self.inference_engine = HailoAsyncInference(
                self.working_model_path,
                self.input_store,
                self.response_store,
                self.batch_size,
            )
            self.input_shape = self.inference_engine.get_input_shape()
            self.validate_model_contract()
            logger.debug(f"[INIT] Model input shape: {self.input_shape}")
            self.inference_thread = threading.Thread(
                target=self.inference_engine.run, daemon=True
            )
            self.inference_thread.start()
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize HailoAsyncInference: {e}")
            raise

    def set_path_and_url(self, path: str = None):
        if not path:
            self.model_path = None
            self.url = None
            return
        if self.is_url(path):
            self.url = path
            self.model_path = None
        else:
            self.model_path = path
            self.url = None

    def is_url(self, url: str) -> bool:
        return (
            url.startswith("http://")
            or url.startswith("https://")
            or url.startswith("www.")
        )

    @staticmethod
    def extract_model_name(path: str = None, url: str = None) -> str:
        if path and path.endswith(".hef"):
            return os.path.basename(path)
        elif url and url.endswith(".hef"):
            return os.path.basename(url)
        else:
            return H10H_DEFAULT_MODEL

    @staticmethod
    def download_model(url: str, destination: str):
        if not url.endswith(".hef"):
            raise ValueError("Invalid model URL. Only .hef files are supported.")
        try:
            urllib.request.urlretrieve(url, destination)
            logger.debug(f"Downloaded model to {destination}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {url}: {str(e)}")

    def check_and_prepare(self) -> str:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        model_name = self.extract_model_name(self.model_path, self.url)
        cached_model_path = os.path.join(self.cache_dir, model_name)
        if not self.model_path and not self.url:
            if os.path.exists(cached_model_path):
                logger.debug(f"Model found in cache: {cached_model_path}")
                return cached_model_path
            else:
                logger.debug(f"Downloading default model: {model_name}")
                self.download_model(H10H_DEFAULT_URL, cached_model_path)
        elif self.url:
            logger.debug(f"Downloading model from URL: {self.url}")
            self.download_model(self.url, cached_model_path)
        elif self.model_path:
            if os.path.exists(self.model_path):
                logger.debug(f"Using existing model at: {self.model_path}")
                return self.model_path
            else:
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        return cached_model_path

    def validate_model_contract(self) -> None:
        model_height, model_width = get_model_hw_from_input_shape(self.input_shape)

        if self.model_height not in (None, model_height):
            raise ValueError(
                f"Configured model height {self.model_height} does not match the HEF input height {model_height}."
            )

        if self.model_width not in (None, model_width):
            raise ValueError(
                f"Configured model width {self.model_width} does not match the HEF input width {model_width}."
            )

    def detect_raw(self, tensor_input):
        tensor_input = self.preprocess(tensor_input)

        if isinstance(tensor_input, np.ndarray) and len(tensor_input.shape) == 3:
            tensor_input = np.expand_dims(tensor_input, axis=0)

        request_id = self.input_store.put(tensor_input)

        try:
            _, infer_results = self.response_store.get(
                request_id, timeout=DEFAULT_INFERENCE_TIMEOUT
            )
        except TimeoutError:
            logger.error(
                f"Timeout waiting for inference results for request {request_id}"
            )

            if not self.inference_thread.is_alive():
                raise RuntimeError(
                    "HailoRT inference thread has stopped, restart required."
                )

            return np.zeros((20, 6), dtype=np.float32)

        if isinstance(infer_results, list) and len(infer_results) == 1:
            infer_results = infer_results[0]
        elif isinstance(infer_results, dict):
            infer_results = list(infer_results.values())

        threshold = 0.4
        all_detections = []
        for class_id, detection_set in enumerate(infer_results):
            if not isinstance(detection_set, np.ndarray) or detection_set.size == 0:
                continue
            for det in detection_set:
                if det.shape[0] < 5:
                    continue
                score = float(det[4])
                if score < threshold:
                    continue
                all_detections.append([class_id, score, det[0], det[1], det[2], det[3]])

        if len(all_detections) == 0:
            detections_array = np.zeros((20, 6), dtype=np.float32)
        else:
            detections_array = np.array(all_detections, dtype=np.float32)
            if detections_array.shape[0] > 20:
                detections_array = detections_array[:20, :]
            elif detections_array.shape[0] < 20:
                pad = np.zeros((20 - detections_array.shape[0], 6), dtype=np.float32)
                detections_array = np.vstack((detections_array, pad))

        return detections_array

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            model_height, model_width = get_model_hw_from_input_shape(self.input_shape)
            processed = preprocess_tensor(image, model_width, model_height)
            return np.expand_dims(processed, axis=0)
        else:
            raise ValueError("Unsupported image format for preprocessing")

    def close(self):
        """Properly shuts down the inference engine and releases the VDevice."""
        logger.debug("[CLOSE] Closing HailoDetector")
        try:
            if hasattr(self, "inference_engine"):
                if hasattr(self.inference_engine, "target"):
                    self.inference_engine.target.release()
                logger.debug("Hailo VDevice released successfully")
        except Exception as e:
            logger.error(f"Failed to close Hailo device: {e}")
            raise

    def __del__(self):
        """Destructor to ensure cleanup when the object is deleted."""
        self.close()


# ----------------- HailoDetectorConfig Class ----------------- #
class HailoDetectorConfig(BaseDetectorConfig):
    """Hailo-10H detector using HEF models and the HailoRT SDK for inference on Hailo hardware."""

    model_config = ConfigDict(
        title="Hailo-10H",
    )

    type: Literal[DETECTOR_KEY]
    device: str = Field(
        default="PCIe",
        title="Device Type",
        description="The device to use for Hailo inference (e.g. 'PCIe', 'M.2').",
    )
