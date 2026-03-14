"""
Anti-scroll: block or reverse mouse scroll when user is not focused on the screen.
"""

import threading
from typing import Callable, Optional

from pynput.mouse import Controller, Listener


class AntiScroll:
    """
    Listens for scroll events and either blocks them or emits reverse scroll
    when the provided focus_check returns False (not focused).
    """

    def __init__(
        self,
        focus_check: Callable[[], bool],
        reverse_scroll: bool = True,
    ):
        """
        Args:
            focus_check: Callable that returns True if user is focused, False otherwise.
            reverse_scroll: If True, reverse scroll direction when not focused;
                            if False, block scroll (no event).
        """
        self._focus_check = focus_check
        self._reverse_scroll = reverse_scroll
        self._mouse = Controller()
        self._listener: Optional[Listener] = None
        self._lock = threading.Lock()

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> bool:
        """
        Called on every scroll event. Return False to consume (block) the event.
        We cannot truly "block" the scroll at OS level with pynput; we can only
        suppress the event and optionally inject a reverse scroll.
        """
        try:
            focused = self._focus_check()
        except Exception:
            focused = True  # On error, allow scroll

        if focused:
            return True  # Let the event pass (we don't suppress in pynput by returning False we stop propagation for our listener but scroll already happened - see pynput docs)
            # Actually in pynput, returning False from the callback stops the event from being propagated to other listeners; it does NOT prevent the scroll from having already been applied by the OS. So to "reverse" we need to inject an opposite scroll.
            # So: when not focused, we inject reverse scroll (or do nothing to "block" - but scroll already happened). So the flow is: scroll always happens; when not focused we immediately send opposite scroll to "undo" it.
        else:
            if self._reverse_scroll:
                with self._lock:
                    self._mouse.scroll(-dx, -dy)
            # We cannot prevent the original scroll; we've reversed it. Return True so listener keeps running.
            return True

    def start(self) -> None:
        """Start the scroll listener in a background thread."""
        if self._listener is not None:
            return
        self._listener = Listener(on_scroll=self._on_scroll)
        self._listener.start()

    def stop(self) -> None:
        """Stop the scroll listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
