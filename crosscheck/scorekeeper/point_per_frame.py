from .base import Scorekeeper


class PointPerFrame(Scorekeeper):

    def __init__(self):
        super().__init__()
        self._total = 0

    def tick(self):
        self._total += 1

        self._done_reasons['long'] = self._total > 300

        return self._total

