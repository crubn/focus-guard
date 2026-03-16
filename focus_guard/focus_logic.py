"""
Focus state logic: maps gaze to focused/not_focused and optional look-away alert.
"""

import time
from typing import Callable, Optional

from focus_guard.utils import (
    FocusState,
    GazeState,
    LOOK_AWAY_ALERT_SECONDS,
    FOCUS_HYSTERESIS_FRAMES,
)


class FocusLogic:
    """
    Maintains focus state from gaze and optionally triggers a callback when
    the user has been looking away for more than LOOK_AWAY_ALERT_SECONDS.
    """

    def __init__(
        self,
        alert_after_seconds: float = LOOK_AWAY_ALERT_SECONDS,
        on_look_away_alert: Optional[Callable[[], None]] = None,
    ):
        self._alert_after_seconds = alert_after_seconds
        self._on_alert = on_look_away_alert
        self._first_not_focused_time: Optional[float] = None
        self._alert_played_this_session: bool = False

        # Гистерезис по фокусу: не переключаемся сразу, требуем N последовательных кадров
        self._current_state: FocusState = FocusState.FOCUSED
        self._hysteresis_counter: int = 0
        self._last_gaze_focused: Optional[bool] = None

    def update(self, gaze: GazeState) -> FocusState:
        """
        Map gaze to focus state with hysteresis. If gaze is not at_screen,
        start timer and optionally call on_look_away_alert once after
        alert_after_seconds.
        """
        is_focused_gaze = gaze == GazeState.AT_SCREEN

        # --- Логика таймера для звукового алерта (по «грубой» оценке — есть/нет фокуса) ---
        if is_focused_gaze:
            self._first_not_focused_time = None
            self._alert_played_this_session = False
        else:
            now = time.monotonic()
            if self._first_not_focused_time is None:
                self._first_not_focused_time = now

            elapsed = now - self._first_not_focused_time
            if (
                self._on_alert
                and not self._alert_played_this_session
                and elapsed >= self._alert_after_seconds
            ):
                self._alert_played_this_session = True
                try:
                    self._on_alert()
                except Exception:
                    pass

        # --- Гистерезис для самого FocusState (минимум N кадров подряд для смены состояния) ---
        if self._last_gaze_focused is None or self._last_gaze_focused == is_focused_gaze:
            # То же самое состояние — накапливаем счётчик
            self._hysteresis_counter += 1
        else:
            # Поменилось направление взгляда — начинаем новый счёт
            self._hysteresis_counter = 1

        self._last_gaze_focused = is_focused_gaze

        target_state = FocusState.FOCUSED if is_focused_gaze else FocusState.NOT_FOCUSED
        if (
            target_state != self._current_state
            and self._hysteresis_counter >= max(1, FOCUS_HYSTERESIS_FRAMES)
        ):
            self._current_state = target_state

        return self._current_state

    def reset_alert(self) -> None:
        """Allow the alert to fire again next time user looks away long enough."""
        self._alert_played_this_session = False
        self._first_not_focused_time = None
