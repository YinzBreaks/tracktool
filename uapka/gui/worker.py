"""Run long pipeline steps off the UI thread, with progress and cancellation."""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    def __init__(self, fn):
        """fn(progress, cancel) -> result"""
        super().__init__()
        self.fn = fn
        self.signals = _Signals()
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            res = self.fn(lambda f, m: self.signals.progress.emit(float(f), str(m)), self._cancel.is_set)
            self.signals.finished.emit(res)
        except InterruptedError:
            self.signals.failed.emit("Cancelled.")
        except Exception as e:  # surfaced in the UI, never swallowed
            self.signals.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


def start(job: Job) -> Job:
    QThreadPool.globalInstance().start(job)
    return job
