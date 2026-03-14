"""
Gaze direction estimation from face landmarks (at screen / down / away).
"""

import numpy as np
from typing import Optional

from focus_guard.eye_detection import FaceLandmarks
from focus_guard.utils import (
    GazeState,
    PITCH_LOOK_DOWN_THRESHOLD,
    PITCH_LOOK_UP_THRESHOLD,
    YAW_LOOK_AWAY_THRESHOLD,
    GAZE_SMOOTHING_ALPHA,
)


class GazeTracker:
    """
    Estimates gaze direction from face landmarks using nose position relative
    to face center (pitch = looking up/down, yaw = looking left/right).
    """

    def __init__(self, smoothing_alpha: float = GAZE_SMOOTHING_ALPHA):
        self._smoothing_alpha = smoothing_alpha
        self._smoothed_pitch: Optional[float] = None
        self._smoothed_yaw: Optional[float] = None

    def update(self, face: Optional[FaceLandmarks]) -> GazeState:
        """
        Update gaze state from current face landmarks. Returns GazeState.
        """
        if face is None:
            # No face: treat as looking away
            return GazeState.LOOKING_AWAY

        nose_x, nose_y = face.nose_tip_normalized()
        cx, cy = face.face_center_normalized()

        # Pitch: nose_y increases when looking down (in image coords, y down)
        # yaw: nose_x offset from center (0.5) = looking left/right
        pitch = nose_y
        yaw = nose_x - 0.5

        # Smooth
        if self._smoothed_pitch is None:
            self._smoothed_pitch = pitch
            self._smoothed_yaw = yaw
        else:
            self._smoothed_pitch = (
                self._smoothing_alpha * pitch + (1 - self._smoothing_alpha) * self._smoothed_pitch
            )
            self._smoothed_yaw = (
                self._smoothing_alpha * yaw + (1 - self._smoothing_alpha) * self._smoothed_yaw
            )

        p = self._smoothed_pitch
        y = self._smoothed_yaw

        # Classify: down (phone) = nose low in frame; away = nose left/right of center
        if p > PITCH_LOOK_DOWN_THRESHOLD:
            return GazeState.LOOKING_DOWN
        if abs(y) > YAW_LOOK_AWAY_THRESHOLD:
            return GazeState.LOOKING_AWAY
        return GazeState.AT_SCREEN

    def reset_smoothing(self) -> None:
        """Reset smoothed values (e.g. when face is lost)."""
        self._smoothed_pitch = None
        self._smoothed_yaw = None
