"""
Shared constants and utility functions for Focus Guard.
"""

import os
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Tuple

# URL официальной модели Face Landmarker (MediaPipe Tasks)
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# -----------------------------------------------------------------------------
# Gaze / focus constants
# -----------------------------------------------------------------------------

# «Смотрю в камеру» = показываем камеру; «смотрю в сторону/вниз» = показываем видео.
# Пороги задают зону «в камеру»: голова не слишком опущена и не сильно повёрнута вбок.
PITCH_LOOK_DOWN_THRESHOLD = 0.58   # nose_y выше = «в камеру», ниже = смотрим вниз (телефон)
PITCH_LOOK_UP_THRESHOLD = 0.45     # (не используется в классификации, оставлен для совместимости)
YAW_LOOK_AWAY_THRESHOLD = 0.38     # |нос - центр| выше = смотрим в сторону

# How much to smooth gaze state (0 = no smoothing, 1 = max smoothing)
GAZE_SMOOTHING_ALPHA = 0.25

# Seconds looking away before playing alert sound
LOOK_AWAY_ALERT_SECONDS = 5.0

# Target FPS for the main loop
TARGET_FPS = 25
FRAME_DELAY_MS = int(1000 / TARGET_FPS)


class GazeState(str, Enum):
    """User gaze direction."""
    AT_SCREEN = "at_screen"
    LOOKING_DOWN = "looking_down"
    LOOKING_AWAY = "looking_away"


class FocusState(str, Enum):
    """High-level focus state for anti-scroll logic."""
    FOCUSED = "focused"
    NOT_FOCUSED = "not_focused"


def get_face_landmarker_model_path() -> str:
    """
    Возвращает путь к файлу модели Face Landmarker (.task).
    При первом запуске скачивает модель из хранилища MediaPipe в папку focus_guard/models/.
    """
    base = Path(__file__).resolve().parent
    model_dir = base / "models"
    model_path = model_dir / "face_landmarker.task"
    if not model_path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
        except Exception as e:
            raise RuntimeError(
                f"Не удалось скачать модель Face Landmarker. "
                f"Скачайте вручную {FACE_LANDMARKER_MODEL_URL} и сохраните как {model_path}. Ошибка: {e}"
            ) from e
    return str(model_path)


def normalize_point(
    point: Tuple[float, float],
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float]:
    """Convert pixel coordinates to normalized [0, 1] (x, y)."""
    if frame_width <= 0 or frame_height <= 0:
        return (0.5, 0.5)
    x = point[0] / frame_width
    y = point[1] / frame_height
    return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
