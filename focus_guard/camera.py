"""
Webcam capture module with error handling.
"""

import cv2
from typing import Optional, Tuple


class CameraError(Exception):
    """Raised when camera cannot be opened or read fails."""
    pass


class Camera:
    """
    Wrapper for OpenCV VideoCapture. Handles open/read errors and provides
    a consistent interface for frame size and release.
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        """
        Args:
            device_index: Webcam deviceindex (0 = default).
            width: Desired frame width.
            height: Desired frame height.
        """
        self._device_index = device_index
        self._requested_width = width
        self._requested_height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the camera. Raises CameraError on failure."""
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Cannot open webcam at index {self._device_index}. "
                "Check that the camera is connected and not in use by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)


