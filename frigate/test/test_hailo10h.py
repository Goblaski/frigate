import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()
