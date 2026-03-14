# Focus Guard

Prevents scrolling on social media (or any app) when you're not looking at your monitor. Uses your webcam and MediaPipe to detect whether you're looking at the screen, down (e.g. at your phone), or away. When you're not focused, scroll events are **reversed** (or optionally only blocked).

## Requirements

- **Python 3.11+**
- Webcam
- macOS / Linux / Windows

## Install

From the project root (parent of `focus_guard/`):

```bash
cd /path/to/CVNUR
pip install -r focus_guard/requirements.txt
```

Or with a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r focus_guard/requirements.txt
```

## Run

From the **repository root** (so `focus_guard` is a package):

```bash
python -m focus_guard.main
```

Or:

```bash
python focus_guard/main.py
```

**Quit:** Press `q` or Escape in the preview window, or close the window.

## Options

| Option | Description |
|--------|-------------|
| `--camera 1` | Use webcam device 1 (default: 0) |
| `--block-only` | Only block scroll when not focused (default: reverse scroll) |
| `--no-alert` | Disable the sound alert after looking away 5 seconds |
| `--scale 1.2` | Scale the preview window (default: 1.0) |

Examples:

```bash
python -m focus_guard.main --no-alert
python -m focus_guard.main --block-only --camera 0
```

## How it works

1. **Webcam** captures frames (640×480 by default).
2. **MediaPipe Face Mesh** detects one face and eye/face landmarks.
3. **Gaze** is estimated from nose position relative to face center: looking **down** (e.g. at phone) or **away** (left/right) vs **at screen**.
4. **Focus state** is either `FOCUSED` (looking at screen) or `NOT_FOCUSED` (down or away).
5. **Anti-scroll**: When not focused, every scroll event is **reversed** (or with `--block-only` only blocked). Reversing effectively undoes the scroll so the page doesn’t move.

## Optional: sound alert

If you look away (or down) for more than **5 seconds**, a short alert plays:

- **Default:** system beep (terminal bell).
- **Custom WAV:** Add `focus_guard/alert.wav` and install `playsound` (`pip install playsound`); the app will play that file instead.

To disable the alert: `python -m focus_guard.main --no-alert`

## Project layout

```
focus_guard/
├── __init__.py
├── main.py           # Entry point, camera loop, debug window
├── camera.py         # Webcam capture (OpenCV)
├── eye_detection.py  # MediaPipe Face Mesh, face/eye landmarks
├── gaze_tracking.py  # Gaze: at_screen / looking_down / looking_away
├── focus_logic.py    # Focus state + optional 5s alert
├── anti_scroll.py    # pynput scroll listener, reverse/block when not focused
├── utils.py          # Constants, enums, helpers
├── requirements.txt
└── README.md
```

## Performance

The pipeline is tuned for **≥20 FPS** (target ~25 FPS) on a typical laptop. You can adjust `TARGET_FPS` and `FRAME_DELAY_MS` in `utils.py` if needed.

## Permissions and environment

- **macOS:** Grant Camera and (if you use pynput) Input Monitoring / Accessibility if the system prompts you. Run from a normal terminal with display access (not over SSH/headless), or MediaPipe may fail with an OpenGL/GPU error.
- **Linux:** Ensure the user has access to the video device (e.g. `video` group).
- **Windows:** Allow camera access when prompted.
