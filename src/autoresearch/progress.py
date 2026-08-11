"""Animated terminal activity for long, otherwise quiet operations."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Sequence
from typing import Self

from rich.console import Console
from rich.status import Status

FUN_PHRASES = (
    "cooking...",
    "thinking in tensors...",
    "turning coffee into forecasts...",
    "herding parallel agents...",
    "reading the residual tea leaves...",
    "chasing lower loss...",
    "keeping the holdout secret...",
    "seasoning the time series...",
    "asking the data nicely...",
    "training tiny models with big dreams...",
)


class Activity:
    """A spinner that rotates playful text while retaining the real phase name."""

    def __init__(
        self,
        console: Console,
        phase: str,
        *,
        interval_s: float = 2.0,
        phrases: Sequence[str] = FUN_PHRASES,
    ) -> None:
        self.console = console
        self.phase = phase
        self.interval_s = interval_s
        self.phrases = tuple(phrases)
        self._started_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: Status | None = None
        self._phrase_index = 0
        self._phrase_order: list[str] = []
        self._stopped = False

    def _reshuffle(self) -> None:
        self._phrase_order = random.sample(self.phrases, len(self.phrases))
        self._phrase_index = 0

    def _next_phrase(self) -> str:
        if not self._phrase_order or self._phrase_index >= len(self._phrase_order):
            self._reshuffle()
        phrase = self._phrase_order[self._phrase_index]
        self._phrase_index += 1
        return phrase

    def _message(self) -> str:
        elapsed = int(time.monotonic() - self._started_at)
        return f"[bold blue]{self.phase}[/bold blue] [dim]— {self._next_phrase()} ({elapsed}s)[/dim]"

    def _animate(self) -> None:
        while not self._stop.wait(self.interval_s):
            if self._status is not None:
                self._status.update(self._message())

    def update(self, phase: str) -> None:
        """Change the factual phase while keeping the same elapsed timer."""
        self.phase = phase
        if self._status is not None and not self._stopped:
            self._status.update(self._message())

    def stop(self) -> None:
        """Stop the animation now; safe to call more than once."""
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 0.1)
        if self._status is not None:
            self._status.stop()

    def __enter__(self) -> Self:
        if not self.console.is_terminal:
            return self
        self._started_at = time.monotonic()
        self._status = self.console.status(self._message(), spinner="dots")
        self._status.start()
        self._thread = threading.Thread(target=self._animate, name="cli-activity", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
