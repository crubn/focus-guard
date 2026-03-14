"""
Focus Guard - Prevents scrolling when user is not looking at the monitor.

Run from project root: python -m focus_guard.main
Or: python focus_guard/main.py (from repo root so focus_guard is a package)
"""

import argparse
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Ensure package is importable when run as script
if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    __package__ = "focus_guard"

from focus_guard.camera import Camera, CameraError
from focus_guard.eye_detection import EyeDetector, FaceLandmarks, are_eyes_closed
from focus_guard.gaze_tracking import GazeTracker
from focus_guard.focus_logic import FocusLogic
from focus_guard.anti_scroll import AntiScroll
from focus_guard.utils import (
    FocusState,
    GazeState,
    FRAME_DELAY_MS,
    LOOK_AWAY_ALERT_SECONDS,
)


# -----------------------------------------------------------------------------
# Optional sound alert
# -----------------------------------------------------------------------------

def play_alert_sound() -> None:
    """Play a short alert when user has been looking away > N seconds."""
    try:
        sound_path = Path(__file__).parent / "alert.wav"
        if sound_path.exists():
            try:
                import playsound
                playsound.playsound(str(sound_path), block=False)
                return
            except ImportError:
                pass
    except Exception:
        pass
    _system_bell()


def _system_bell() -> None:
    """Fallback: system beep (works on most OS)."""
    try:
        import os
        # ASCII bell
        print("\a", end="", flush=True)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# "Look at screen" reminder — видео или картинка (при отводе взгляда и при старте)
# -----------------------------------------------------------------------------

REMINDER_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm")

LOOK_AT_SCREEN_IMAGE_URL = (
    "https://images.steamusercontent.com/ugc/18088450126317235132/870B19119AB711D3E763CB8F0E18F2153E8A1BF0/"
    "?imw=512&ima=fit&impolicy=Letterbox&imcolor=%23000000&letterbox=false"
)


def get_reminder_media_dir() -> Path:
    """Папка, где лежат look_at_screen.png / look_at_screen.mp4."""
    return Path(__file__).resolve().parent


def get_reminder_video_path() -> Path | None:
    """Путь к видео «смотри в экран», если есть файл look_at_screen.* с подходящим расширением."""
    base = get_reminder_media_dir() / "look_at_screen"
    for ext in REMINDER_VIDEO_EXTENSIONS:
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None


def open_reminder_video(path: Path) -> cv2.VideoCapture | None:
    """Открывает видео для воспроизведения по кругу. Возвращает VideoCapture или None."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    return cap


def get_look_at_screen_image_path() -> Path:
    """Путь к локальной копии картинки «смотри в экран»."""
    return get_reminder_media_dir() / "look_at_screen.png"


def ensure_look_at_screen_image() -> Path:
    """Скачивает картинку по URL, если файла ещё нет. Возвращает путь к файлу."""
    path = get_look_at_screen_image_path()
    if path.exists():
        return path
    try:
        req = urllib.request.Request(
            LOOK_AT_SCREEN_IMAGE_URL,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        if data[:100].strip().lower().startswith(b"<"):
            raise ValueError("Сервер вернул HTML вместо картинки")
        path.write_bytes(data)
    except Exception as e:
        print(f"Картинку по ссылке скачать не удалось: {e}", file=sys.stderr)
        print(f"  Сохраните картинку вручную как: {path}", file=sys.stderr)
    return path


def load_reminder_image(frame_width: int, frame_height: int):
    """
    Загружает картинку «смотри в экран» и подгоняет под размер кадра.
    Возвращает BGR-кадр того же размера или None при ошибке.
    """
    path = ensure_look_at_screen_image()
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    return cv2.resize(img, (frame_width, frame_height), interpolation=cv2.INTER_AREA)


def get_video_duration_ms(cap: cv2.VideoCapture) -> float:
    """Длительность видео в миллисекундах (для синхронизации со звуком)."""
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    if count is None or count <= 0:
        return 0.0
    return (count / fps) * 1000.0


def read_reminder_video_frame_next(cap: cv2.VideoCapture, frame_width: int, frame_height: int) -> np.ndarray | None:
    """Читает следующий кадр; в конце перематывает в начало (без синхронизации по времени)."""
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)


def read_reminder_video_frame_synced(
    cap: cv2.VideoCapture,
    frame_width: int,
    frame_height: int,
    start_time: float,
    duration_ms: float,
) -> np.ndarray | None:
    """
    Читает кадр видео по текущему времени воспроизведения (как у звука).
    start_time = time.perf_counter() в момент старта, duration_ms — длина ролика в мс.
    """
    if duration_ms <= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        position_ms = elapsed_ms % duration_ms
        cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
        ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)


# -----------------------------------------------------------------------------
# Звук к видео-напоминанию (afplay на macOS, по кругу пока показываем видео)
# -----------------------------------------------------------------------------

def _reminder_audio_loop(path: Path, stop_event: threading.Event, process_holder: list) -> None:
    """В фоне крутит afplay по кругу, пока не выставлен stop_event."""
    import platform
    while not stop_event.is_set():
        if platform.system() == "Darwin":
            cmd = ["afplay", str(path)]
        else:
            # Linux: paplay/ffplay; Windows: не поддерживается без доп. библиотек
            break
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process_holder[0] = p
            p.wait()
        except Exception:
            pass
        finally:
            process_holder[0] = None
        if stop_event.is_set():
            break


def start_reminder_audio(path: Path) -> tuple[threading.Event, list, threading.Thread | None]:
    """Запускает воспроизведение звука к видео в фоне. Возвращает (stop_event, process_holder, thread)."""
    stop_event = threading.Event()
    process_holder = [None]
    thread = threading.Thread(target=_reminder_audio_loop, args=(path, stop_event, process_holder), daemon=True)
    thread.start()
    return stop_event, process_holder, thread


def stop_reminder_audio(stop_event: threading.Event, process_holder: list, thread: threading.Thread | None) -> None:
    """Останавливает звук и ждёт завершения потока."""
    stop_event.set()
    if process_holder[0] is not None:
        try:
            process_holder[0].terminate()
            process_holder[0].wait(timeout=2)
        except Exception:
            pass
    if thread is not None:
        thread.join(timeout=2)


# -----------------------------------------------------------------------------
# Debug overlay
# -----------------------------------------------------------------------------

def draw_landmarks(frame, face: FaceLandmarks) -> None:
    """Draw face/eye landmarks on frame."""
    from focus_guard.eye_detection import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, NOSE_TIP_IDX
    h, w = frame.shape[:2]
    for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES + [NOSE_TIP_IDX]:
        x, y = face.get_pixel(idx)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    # Nose tip slightly larger
    nx, ny = face.get_pixel(NOSE_TIP_IDX)
    cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)


def draw_overlay(
    frame,
    gaze: GazeState,
    focus: FocusState,
    fps: float,
) -> None:
    """Draw gaze, focus label and FPS."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    # FOCUSED / NOT FOCUSED
    if focus == FocusState.FOCUSED:
        label, color = "FOCUSED", (0, 255, 0)
    else:
        label, color = "NOT FOCUSED", (0, 0, 255)
    cv2.putText(frame, label, (10, y_offset), font, 0.8, color, 2)
    y_offset += 28
    cv2.putText(frame, f"Gaze: {gaze.value}", (10, y_offset), font, 0.5, (255, 255, 255), 1)
    y_offset += 22
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), font, 0.5, (200, 200, 200), 1)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def run(
    camera_index: int = 0,
    reverse_scroll: bool = True,
    enable_alert: bool = True,
    window_scale: float = 1.0,
) -> None:
    """Run the focus guard loop."""
    # Thread-safe focus state for anti-scroll
    focus_state: FocusState = FocusState.FOCUSED
    state_lock = threading.Lock()

    def get_focused() -> bool:
        with state_lock:
            return focus_state == FocusState.FOCUSED

    anti_scroll = AntiScroll(focus_check=get_focused, reverse_scroll=reverse_scroll)
    anti_scroll.start()

    alert_callback = play_alert_sound if enable_alert else None
    focus_logic = FocusLogic(
        alert_after_seconds=LOOK_AWAY_ALERT_SECONDS,
        on_look_away_alert=alert_callback,
    )

    camera = Camera(device_index=camera_index, width=640, height=480)
    try:
        camera.open()
    except CameraError as e:
        print(f"Camera error: {e}", file=sys.stderr)
        anti_scroll.stop()
        sys.exit(1)

    eye_detector = None
    reminder_video_cap = None
    was_playing_reminder_audio = False
    reminder_audio_stop_event = None
    reminder_audio_process_holder = None
    reminder_audio_thread = None
    reminder_video_start_time = 0.0
    reminder_video_duration_ms = 0.0
    try:
        eye_detector = EyeDetector()
        gaze_tracker = GazeTracker()

        window_name = "Focus Guard - Press 'q' to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if window_scale != 1.0:
            cv2.resizeWindow(window_name, int(640 * window_scale), int(480 * window_scale))

        fps_frame_count = 0
        fps_start = time.perf_counter()
        fps_value = 0.0
        frame_w, frame_h = camera.get_size()
        # Видео имеет приоритет над картинкой: если есть look_at_screen.mp4 (.mov и т.д.) — играет оно
        reminder_video_cap = None
        reminder_video_path = get_reminder_video_path()
        if reminder_video_path:
            reminder_video_cap = open_reminder_video(reminder_video_path)
            if reminder_video_cap is not None:
                reminder_video_duration_ms = get_video_duration_ms(reminder_video_cap)
        reminder_image = load_reminder_image(frame_w, frame_h) if reminder_video_cap is None else None
        run_start = time.perf_counter()
        show_reminder_seconds = 3.0  # первые N секунд всегда показывать напоминание при старте
        eyes_closed_seconds = 3.0  # если глаза закрыты дольше — показываем напоминание
        eyes_closed_start_time = None
        eyes_closed_trigger = False

        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            face = eye_detector.process(frame)
            gaze = gaze_tracker.update(face)
            focus = focus_logic.update(gaze)

            # Закрыты ли глаза дольше N секунд — тогда тоже показываем напоминание
            if are_eyes_closed(face):
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.perf_counter()
                elif (time.perf_counter() - eyes_closed_start_time) >= eyes_closed_seconds:
                    eyes_closed_trigger = True
            else:
                eyes_closed_start_time = None
                eyes_closed_trigger = False

            with state_lock:
                focus_state = focus

            # Смотрю в камеру → камера; смотрю в сторону/вниз/глаза закрыты 3 сек → видео-напоминание
            just_started = (time.perf_counter() - run_start) < show_reminder_seconds
            show_reminder = (
                just_started
                or (focus == FocusState.NOT_FOCUSED)  # отвод взгляда или вниз
                or eyes_closed_trigger
            )

            # Звук к видео: старт при показе видео, стоп при возврате к камере
            if show_reminder and reminder_video_cap is not None and reminder_video_path is not None:
                if not was_playing_reminder_audio:
                    reminder_video_start_time = time.perf_counter()
                    reminder_audio_stop_event, reminder_audio_process_holder, reminder_audio_thread = start_reminder_audio(reminder_video_path)
                    was_playing_reminder_audio = True
            else:
                if was_playing_reminder_audio:
                    stop_reminder_audio(reminder_audio_stop_event, reminder_audio_process_holder, reminder_audio_thread)
                    was_playing_reminder_audio = False

            if show_reminder:
                if reminder_video_cap is not None and reminder_video_duration_ms > 0:
                    display_frame = read_reminder_video_frame_synced(
                        reminder_video_cap, frame_w, frame_h,
                        reminder_video_start_time, reminder_video_duration_ms,
                    )
                elif reminder_video_cap is not None:
                    display_frame = read_reminder_video_frame_next(reminder_video_cap, frame_w, frame_h)
                else:
                    display_frame = None
                if reminder_video_cap is None or display_frame is None:
                    if reminder_image is not None:
                        display_frame = reminder_image.copy()
                    else:
                        display_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                        display_frame[:] = (40, 40, 40)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display_frame, "Look at the screen!", (10, 30), font, 0.8, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Gaze: {gaze.value}", (10, 58), font, 0.5, (255, 255, 255), 1)
            else:
                display_frame = frame
                if face is not None:
                    draw_landmarks(display_frame, face)
                # Rolling FPS
                fps_frame_count += 1
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 0.5:
                    fps_value = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start = time.perf_counter()
                draw_overlay(display_frame, gaze, focus, fps_value)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(max(1, FRAME_DELAY_MS))
            if key == ord("q") or key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        if was_playing_reminder_audio and reminder_audio_stop_event is not None:
            stop_reminder_audio(reminder_audio_stop_event, reminder_audio_process_holder, reminder_audio_thread)
        cv2.destroyAllWindows()
        camera.release()
        if reminder_video_cap is not None:
            reminder_video_cap.release()
        anti_scroll.stop()
        if eye_detector is not None:
            eye_detector.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focus Guard: block/reverse scroll when not looking at the screen."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--block-only",
        action="store_true",
        help="Block scroll when not focused (default: reverse scroll)",
    )
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Disable sound alert after looking away 5 seconds",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Preview window scale (default: 1.0)",
    )
    args = parser.parse_args()
    run(
        camera_index=args.camera,
        reverse_scroll=not args.block_only,
        enable_alert=not args.no_alert,
        window_scale=args.scale,
    )


if __name__ == "__main__":
    main()
