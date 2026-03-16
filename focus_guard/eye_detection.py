"""
Face and eye landmark detection using MediaPipe Face Landmarker (Tasks API).
Supports both legacy mp.solutions.face_mesh and new mediapipe.tasks.vision API.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from focus_guard.utils import get_face_landmarker_model_path

# MediaPipe Face Landmarker uses the same landmark indices as Face Mesh
# (see https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
NOSE_TIP_IDX = 4
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 387, 385, 263, 373, 380]
# Eye Aspect Ratio (EAR): 6 точек на глаз — вертикали (p2-p6, p3-p5) и горизонталь (p1-p4)
LEFT_EAR_INDICES = (33, 160, 158, 133, 153, 144)   # p1, p2, p3, p4, p5, p6
RIGHT_EAR_INDICES = (362, 385, 387, 263, 373, 380)
EAR_CLOSED_THRESHOLD = 0.22  # ниже — глаз считаем закрытым (поднять до 0.25, если не срабатывает)
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


class FaceLandmarks:
    """Holds normalized face landmarks and frame dimensions."""

    def __init__(
        self,
        landmarks: List[Tuple[float, float, float]],
        frame_width: int,
        frame_height: int,
    ):
        self._landmarks = landmarks
        self._w = frame_width
        self._h = frame_height

    def get_normalized(self, index: int) -> Tuple[float, float]:
        if index < 0 or index >= len(self._landmarks):
            return (0.5, 0.5)
        x, y, _ = self._landmarks[index]
        return (float(x), float(y))

    def get_pixel(self, index: int) -> Tuple[int, int]:
        x, y = self.get_normalized(index)
        return (int(x * self._w), int(y * self._h))

    def nose_tip_normalized(self) -> Tuple[float, float]:
        return self.get_normalized(NOSE_TIP_IDX)

    def face_center_normalized(self) -> Tuple[float, float]:
        nx, ny = self.get_normalized(NOSE_TIP_IDX)
        left_x = np.mean([self.get_normalized(i)[0] for i in LEFT_EYE_INDICES])
        left_y = np.mean([self.get_normalized(i)[1] for i in LEFT_EYE_INDICES])
        right_x = np.mean([self.get_normalized(i)[0] for i in RIGHT_EYE_INDICES])
        right_y = np.mean([self.get_normalized(i)[1] for i in RIGHT_EYE_INDICES])
        cx = (left_x + right_x) / 2
        cy = (left_y + right_y + ny) / 3
        return (float(cx), float(cy))

    def left_eye_center(self) -> Tuple[float, float]:
        xs = [self.get_normalized(i)[0] for i in LEFT_EYE_INDICES]
        ys = [self.get_normalized(i)[1] for i in LEFT_EYE_INDICES]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def right_eye_center(self) -> Tuple[float, float]:
        xs = [self.get_normalized(i)[0] for i in RIGHT_EYE_INDICES]
        ys = [self.get_normalized(i)[1] for i in RIGHT_EYE_INDICES]
        return (float(np.mean(xs)), float(np.mean(ys)))

    @property
    def frame_width(self) -> int:
        return self._w

    @property
    def frame_height(self) -> int:
        return self._h


def _eye_aspect_ratio(face: FaceLandmarks, p1: int, p2: int, p3: int, p4: int, p5: int, p6: int) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|). Норма ~0.25–0.3, при закрытии падает."""
    def dist(a: int, b: int) -> float:
        x1, y1 = face.get_normalized(a)
        x2, y2 = face.get_normalized(b)
        return float(np.hypot(x2 - x1, y2 - y1))
    v1 = dist(p2, p6)
    v2 = dist(p3, p5)
    h = dist(p1, p4)
    if h <= 1e-6:
        return 0.5
    return (v1 + v2) / (2.0 * h)


def are_eyes_closed(face: Optional[FaceLandmarks], threshold: float = EAR_CLOSED_THRESHOLD) -> bool:
    """True, если оба глаза закрыты (EAR ниже порога)."""
    if face is None:
        return False
    left_ear = _eye_aspect_ratio(face, *LEFT_EAR_INDICES)
    right_ear = _eye_aspect_ratio(face, *RIGHT_EAR_INDICES)
    return left_ear < threshold and right_ear < threshold


def _create_detector_tasks_api(model_path: str, num_faces: int, min_confidence: float):
    """Create FaceLandmarker using MediaPipe Tasks API (mediapipe >= 0.10.31)."""
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import face_landmarker as face_landmarker_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode

    BaseOptions = base_options_lib.BaseOptions
    FaceLandmarker = face_landmarker_lib.FaceLandmarker
    FaceLandmarkerOptions = face_landmarker_lib.FaceLandmarkerOptions
    VisionRunningMode = vision_task_running_mode.VisionTaskRunningMode

    # CPU delegate avoids OpenGL/GPU errors on macOS and headless environments
    base_opts = BaseOptions(
        model_asset_path=model_path,
        delegate=BaseOptions.Delegate.CPU,
    )
    options = FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionRunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    return FaceLandmarker.create_from_options(options)


def _create_detector_legacy_api():
    """Create FaceMesh using legacy mp.solutions (mediapipe < 0.10.31)."""
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class EyeDetector:
    """
    Face and eye landmark detector. Uses MediaPipe Tasks API (FaceLandmarker)
    when available, otherwise legacy Face Mesh.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._min_confidence = min_detection_confidence
        self._landmarker = None
        self._legacy_face_mesh = None
        self._use_tasks_api = None
        self._video_timestamp_ms = 0

        # Try new Tasks API first
        try:
            model_path = get_face_landmarker_model_path()
            self._landmarker = _create_detector_tasks_api(
                model_path, max_num_faces, min_detection_confidence
            )
            self._use_tasks_api = True
        except Exception:
            # Fallback to legacy mp.solutions.face_mesh
            try:
                self._legacy_face_mesh = _create_detector_legacy_api()
                self._use_tasks_api = False
            except Exception as e:
                err = str(e)
                hint = ""
                if "NSOpenGLPixelFormat" in err or "kGpuService" in err:
                    hint = " Запустите приложение из обычного терминала с доступом к дисплею (не в headless/SSH)."
                raise RuntimeError(
                    "Не удалось инициализировать детектор лиц (MediaPipe Tasks API)."
                    f"{hint} Ошибка: {e}"
                ) from e

    def process(self, frame: "cv2.typing.MatLike") -> Optional[FaceLandmarks]:
        """Run face landmark detection. Returns FaceLandmarks for the first face or None."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._use_tasks_api:
            return self._process_tasks_api(rgb, w, h)
        return self._process_legacy(rgb, w, h)

    def _process_tasks_api(self, rgb: np.ndarray, w: int, h: int) -> Optional[FaceLandmarks]:
        from mediapipe.tasks.python.vision.core import image as image_lib

        mp_image = image_lib.Image(
            image_format=image_lib.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self._landmarker.detect_for_video(mp_image, self._video_timestamp_ms)
        self._video_timestamp_ms += 40  # ~25 FPS

        if not result.face_landmarks:
            return None
        first_face = result.face_landmarks[0]
        landmarks = [(float(lm.x), float(lm.y), float(lm.z)) for lm in first_face]
        return FaceLandmarks(landmarks, w, h)

    def _process_legacy(self, rgb: np.ndarray, w: int, h: int) -> Optional[FaceLandmarks]:
        results = self._legacy_face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        raw = results.multi_face_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in raw.landmark]
        return FaceLandmarks(landmarks, w, h)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._legacy_face_mesh is not None:
            self._legacy_face_mesh.close()
            self._legacy_face_mesh = None
