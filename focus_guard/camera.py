"""
Webcam capture module with error handling.
"""

import cv2
from typing import Optional, Tuple


class CameraError(Exception):
    """Raised when camera cannot be opened or read fails."""
    pass





