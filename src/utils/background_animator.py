import threading
import time

from streamlit.runtime.scriptrunner import add_script_run_ctx

from src.pages.prediction.flow.progress_reporter import ProgressReporter


class BackgroundAnimator:
    def __init__(
        self,
        reporter: ProgressReporter,
        text: str,
        interval: float = 0.5,
    ):
        self.reporter = reporter
        self.text = text
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.reporter.emit(self.text, force=True)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate_loop, daemon=True)

        if add_script_run_ctx:
            add_script_run_ctx(self._thread)

        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        return False

    def _animate_loop(self):
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            if self._stop_event.is_set():
                break
            try:
                if self.reporter.last_text == self.text:
                    self.reporter.emit(self.text, force=True)
            except Exception:
                pass
