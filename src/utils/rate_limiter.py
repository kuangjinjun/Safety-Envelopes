import time
import threading

class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.lock = threading.Lock()
        if requests_per_second <= 0:
            self.delay = 0
        else:
            self.delay = 1.0 / requests_per_second
        self.last_call_time = 0

    def acquire(self):
        if self.delay == 0:
            return
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call_time
            wait_time = self.delay - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call_time = time.monotonic()