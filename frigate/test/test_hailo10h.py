import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from pydantic import parse_obj_as

from frigate.config import DetectorConfig
from frigate.detectors.plugins.hailo10h import (
    DEFAULT_INFERENCE_TIMEOUT,
    HAILO10H_ARCH_CAPS,
    SHARED_VDEVICE_GROUP_ID,
    HailoDeviceInfo,
    HailoDetector,
    detect_hailo_device_info,
    get_model_hw_from_input_shape,
    validate_hailo10h_device,
)


DEBUG_VIDEO_PATH = Path(__file__).resolve().parents[2] / "debug" / "car-stopping.mp4"
DEBUG_RESULTS_PATH = DEBUG_VIDEO_PATH.with_name("car-stopping.hailo10h.json")


def get_hailo10h_runtime_error() -> str | None:
    try:
        detect_hailo_device_info()
        return None
    except RuntimeError as exc:
        return str(exc)


def run_hailo10h_detector_on_video(video_path: Path, output_path: Path) -> None:
    cfg_payload = {
        "type": "hailo10h",
        "device": os.environ.get("FRIGATE_HAILO10H_DEVICE", "PCIe"),
        "model": {
            "width": int(os.environ.get("FRIGATE_HAILO10H_MODEL_WIDTH", "320")),
            "height": int(os.environ.get("FRIGATE_HAILO10H_MODEL_HEIGHT", "320")),
        },
    }

    model_path = os.environ.get("FRIGATE_HAILO10H_MODEL_PATH")
    if model_path:
        cfg_payload["model"]["path"] = model_path

    detector = HailoDetector(parse_obj_as(DetectorConfig, cfg_payload))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"Unable to open debug video at {video_path}")

    max_frames = int(os.environ.get("FRIGATE_HAILO10H_VIDEO_MAX_FRAMES", "0"))
    frames = []

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw = detector.detect_raw(frame)
            detections = []
            for det in raw.tolist():
                if len(det) != 6 or det[1] <= 0:
                    continue
                detections.append(
                    {
                        "class_id": int(det[0]),
                        "score": float(det[1]),
                        "bbox": [
                            float(det[2]),
                            float(det[3]),
                            float(det[4]),
                            float(det[5]),
                        ],
                    }
                )

            frames.append({"frame_index": frame_index, "detections": detections})
            frame_index += 1

            if max_frames and frame_index >= max_frames:
                break
    finally:
        cap.release()
        detector.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "frames_processed": len(frames),
                "results": frames,
            },
            indent=2,
        )
    )


class TestHailo10HHelpers(unittest.TestCase):
    @patch("frigate.detectors.plugins.hailo10h.subprocess.run")
    def test_detect_hailo_device_info_parses_identify_output(self, mock_run):
        mock_run.return_value = SimpleNamespace(
            returncode=0,
            stdout=(
                "Firmware Version: 5.1.1 (release,app)\n"
                "Device Architecture: HAILO10H\n"
            ),
            stderr="",
        )

        info = detect_hailo_device_info()

        self.assertEqual(info.architecture, "HAILO10H")
        self.assertEqual(info.firmware_version, "5.1.1 (release,app)")

    def test_validate_hailo10h_device_rejects_non_h10h(self):
        with self.assertRaisesRegex(RuntimeError, "requires a Hailo-10H"):
            validate_hailo10h_device(
                HailoDeviceInfo(
                    architecture="HAILO8L",
                    firmware_version="4.21.0",
                    raw_output="",
                )
            )

    def test_get_model_hw_from_input_shape_rejects_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "Expected a 3D tensor"):
            get_model_hw_from_input_shape((1, 320, 320, 3))

        with self.assertRaisesRegex(ValueError, "Expected 3 channels"):
            get_model_hw_from_input_shape((320, 320, 1))

        self.assertEqual(get_model_hw_from_input_shape((320, 320, 3)), (320, 320))


class TestHailo10HUpstreamAlignment(unittest.TestCase):
    def test_validate_hailo10h_device_uses_upstream_arch_constant(self):
        with self.assertRaisesRegex(RuntimeError, HAILO10H_ARCH_CAPS):
            validate_hailo10h_device(
                HailoDeviceInfo(
                    architecture="HAILO8",
                    firmware_version="4.23.0",
                    raw_output="",
                )
            )

    def test_shared_vdevice_group_constant_matches_hailo_apps(self):
        self.assertEqual(SHARED_VDEVICE_GROUP_ID, "SHARED")


class TestHailo10HDetector(unittest.TestCase):
    @patch("frigate.detectors.plugins.hailo10h.logger.error")
    @patch("frigate.detectors.plugins.hailo10h.detect_hailo_device_info")
    @patch("frigate.detectors.plugins.hailo10h.HailoAsyncInference")
    def test_init_rejects_config_dimensions_that_do_not_match_hef(
        self, mock_engine_cls, mock_device_info, mock_logger_error
    ):
        mock_device_info.return_value = HailoDeviceInfo(
            architecture="HAILO10H",
            firmware_version="5.1.1 (release,app)",
            raw_output="",
        )
        mock_engine = Mock()
        mock_engine.get_input_shape.return_value = (320, 320, 3)
        mock_engine_cls.return_value = mock_engine

        cfg = parse_obj_as(
            DetectorConfig,
            {
                "type": "hailo10h",
                "device": "PCIe",
                "model": {
                    "path": "/etc/hosts",
                    "width": 640,
                    "height": 320,
                },
            },
        )

        with self.assertRaisesRegex(ValueError, "Configured model width 640"):
            HailoDetector(cfg)

        mock_logger_error.assert_not_called()

    def test_detect_raw_accepts_dict_outputs(self):
        detector = object.__new__(HailoDetector)
        detector.input_shape = (320, 320, 3)
        detector.input_store = Mock()
        detector.input_store.put.return_value = 123
        detector.response_store = Mock()
        detector.response_store.get.return_value = (
            None,
            {
                "nms": np.array(
                    [
                        [0.1, 0.2, 0.3, 0.4, 0.95],
                        [0.2, 0.3, 0.4, 0.5, 0.20],
                    ],
                    dtype=np.float32,
                )
            },
        )
        detector.inference_thread = Mock()
        detector.inference_thread.is_alive.return_value = True
        detector.preprocess = Mock(
            return_value=np.zeros((1, 320, 320, 3), dtype=np.uint8)
        )

        result = HailoDetector.detect_raw(detector, np.zeros((320, 320, 3), np.uint8))

        detector.response_store.get.assert_called_once_with(
            123, timeout=DEFAULT_INFERENCE_TIMEOUT
        )
        self.assertEqual(result.shape, (20, 6))
        self.assertEqual(result[0][0], 0)
        self.assertAlmostEqual(float(result[0][1]), 0.95, places=5)


class TestHailo10HVideoHelpers(unittest.TestCase):
    @patch.dict(os.environ, {"FRIGATE_HAILO10H_VIDEO_MAX_FRAMES": "2"}, clear=False)
    @patch("frigate.test.test_hailo10h.cv2.VideoCapture")
    @patch("frigate.test.test_hailo10h.HailoDetector")
    def test_run_hailo10h_detector_on_video_writes_results(
        self, mock_detector_cls, mock_video_capture
    ):
        mock_detector = Mock()
        mock_detector.detect_raw.side_effect = [
            np.array([[1, 0.9, 0.1, 0.2, 0.3, 0.4]] + [[0, 0, 0, 0, 0, 0]] * 19),
            np.zeros((20, 6), dtype=np.float32),
        ]
        mock_detector_cls.return_value = mock_detector

        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((320, 320, 3), dtype=np.uint8)),
            (True, np.zeros((320, 320, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_video_capture.return_value = mock_cap

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "car-stopping.mp4"
            video_path.write_bytes(b"fake")
            output_path = Path(tmpdir) / "car-stopping.hailo10h.json"

            run_hailo10h_detector_on_video(video_path, output_path)

            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["video"], str(video_path))
            self.assertEqual(payload["frames_processed"], 2)
            self.assertEqual(payload["results"][0]["detections"][0]["class_id"], 1)
            mock_detector.close.assert_called_once()
            mock_cap.release.assert_called_once()


class TestHailo10HVideo(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("FRIGATE_RUN_HAILO10H_VIDEO_TEST") == "1",
        "Set FRIGATE_RUN_HAILO10H_VIDEO_TEST=1 to run the Hailo-10H debug video test.",
    )
    def test_run_detector_on_debug_video_and_store_results(self):
        if not DEBUG_VIDEO_PATH.exists():
            self.skipTest(f"Debug video not found: {DEBUG_VIDEO_PATH}")

        runtime_error = get_hailo10h_runtime_error()
        if runtime_error is not None:
            self.skipTest(f"Hailo-10H runtime unavailable: {runtime_error}")

        run_hailo10h_detector_on_video(DEBUG_VIDEO_PATH, DEBUG_RESULTS_PATH)

        self.assertTrue(DEBUG_RESULTS_PATH.exists())
        payload = json.loads(DEBUG_RESULTS_PATH.read_text())
        self.assertEqual(payload["video"], str(DEBUG_VIDEO_PATH))
        self.assertGreater(payload["frames_processed"], 0)
        self.assertIn("results", payload)


if __name__ == "__main__":
    unittest.main()
