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

# «Смотрю в камеру» = камера; «смотрю в сторону/вниз/глаза закрыты» = видео.
# Если у тебя наоборот (при взгляде в камеру играет видео) — поставь INVERT_GAZE = True.
INVERT_GAZE = True

PITCH_LOOK_DOWN_THRESHOLD = 0.55   # нос ниже = смотрим вниз (чуть чувствительнее)
PITCH_LOOK_UP_THRESHOLD = 0.45
YAW_LOOK_AWAY_THRESHOLD = 0.35     # нос в сторону от центра (чуть чувствительнее)

# Сглаживание взгляда: больше = быстрее реакция (0.55 ≈ 2–3 кадра), меньше = плавнее но медленнее
GAZE_SMOOTHING_ALPHA = 0.55

# Сколько кадров подряд нужно для смены фокуса (1 = без гистерезиса, 2 = меньше мерцания)
FOCUS_HYSTERESIS_FRAMES = 2

# Сколько кадров подряд «глаза закрыты» перед стартом таймера 3 сек (защита от моргания)
EYES_CLOSED_CONSECUTIVE_FRAMES = 2

# Seconds looking away before playing alert sound
LOOK_AWAY_ALERT_SECONDS = 5.0

# Target FPS for the main loop (выше = отзывчивее)
TARGET_FPS = 30
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
