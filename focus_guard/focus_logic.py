"""
Focus state logic: maps gaze to focused/not_focused and optional look-away alert.
"""

import time
from typing import Callable, Optional

from focus_guard.utils import FocusState, GazeState, LOOK_AWAY_ALERT_SECONDS


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

    def update(self, gaze: GazeState) -> FocusState:
        """
        Map gaze to focus state. If gaze is not at_screen, start timer and
        optionally call on_look_away_alert once after alert_after_seconds.
        """
        if gaze == GazeState.AT_SCREEN:
            self._first_not_focused_time = None
            self._alert_played_this_session = False
            return FocusState.FOCUSED

        # Not focused
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

        return FocusState.NOT_FOCUSED

    def reset_alert(self) -> None:
        """Allow the alert to fire again next time user looks away long enough."""
        self._alert_played_this_session = False
        self._first_not_focused_time = None
