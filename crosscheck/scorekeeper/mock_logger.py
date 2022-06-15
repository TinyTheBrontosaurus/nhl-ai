from .base import Scorekeeper
from crosscheck.log_folder import LogFolder
from crosscheck.player.utils import RisingEdge

import json

class MockLogger(Scorekeeper):
    """
    This scorekeeper simply logs state
    """

    def __init__(self):
        super().__init__()
        self._total = 0
        self._filename = None
        self._stoppage_count = 0
        self._is_playing = False
        self._last_time = 0
        self._stoppage_detector = RisingEdge()

    def _tick(self):
        if self._filename is None:
            self._filename = LogFolder.folder / "mocklogger.jsonl"
        self._total += 1

        # Figure out if we're playing. it will have a short delay
        if self._total % 24 == 0:
            self._is_playing = self.info['time'] == self._last_time - 1
            self._last_time = self.info['time']
            if self._stoppage_detector.update(not self._is_playing):
                self._stoppage_count += 1

        self._stats["is_playing"] = self._is_playing
        self._stats["stoppage_count"] = self._stoppage_count

        with self._filename.open("a+") as f:
            f.write("{}\n".format(json.dumps(self.info)))

        return self._total

    @classmethod
    def fitness_threshold(cls) -> float:
        """
        Accessor fitness threshold (the score over
        which to stop training)
        """
        return 1e6  # Nothing relevant about this number
