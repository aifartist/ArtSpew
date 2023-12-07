import time
import logging


class Timer:
    def __init__(self, name=None):
        self.name = name
        self._start_time = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def start(self):
        self._start_time = time.time()

    def elapsed(self):
        if self._start_time is None:
            raise RuntimeError("Timer not started. Call start() to start the Timer.")
        elapsed = time.time() - self._start_time
        return self._format_time(elapsed)

    @staticmethod
    def _format_time(elapsed):
        if elapsed < 1:
            return f"{elapsed * 1000:.2f}ms"
        if elapsed < 60:
            return f"{elapsed:.2f}s"
        if elapsed < 3600:
            return f"{elapsed / 60:.1f}m"
        return f"{elapsed / 3600:.1f}h"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = self.elapsed()
        if self.name:
            self._logger.info(f"{elapsed} {self.name}")
        else:
            self._logger.info(f"Elapsed time: {elapsed}")
