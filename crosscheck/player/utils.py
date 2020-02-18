import time
from loguru import logger


class RateController:

    def __init__(self, time_per_iter):
        self.time_per_iter = time_per_iter
        self._next_time = self.reinit()

    def reinit(self):
        self._next_time = time.time() + self.time_per_iter
        return self._next_time

    def tick(self):
        now = time.time()
        delay_needed = self._next_time - now
        if delay_needed > 0:
            time.sleep(delay_needed)
        else:
            logger.warning(f"Falling behind {-delay_needed:.3f}s")
            next_time = now
        self._next_time += self.time_per_iter


class RisingEdge:
    def __init__(self):
        self._request_high = True
        self.state = False

    def update(self, request):
        # Return True only on a rising edge
        self.state = False

        if not self._request_high and request:
            self.state = True
            self._request_high = True
        self._request_high = request

        return self.state

    def reset(self):
        self._request_high = True
        self.state = False
