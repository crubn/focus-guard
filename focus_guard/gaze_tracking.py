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
    INVERT_GAZE,
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
            # Нет лица — считаем «в сторону» и сбрасываем сглаживание, чтобы при появлении лица реакция была быстрой
            self.reset_smoothing()
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

        # Classify: down = нос низко; away = нос в сторону от центра


