from .base import Scorekeeper
from crosscheck.log_folder import LogFolder
from crosscheck.player.utils import RisingEdge

import json


def enum_to_str(enum):
    if enum == 0:
        return "demo"
    elif enum == 1:
        return "home"
    elif enum == 2:
        return "away"
    elif enum == 3:
        return "both"
    else:
        return "error"

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

        # Figure out if we're playing. it will have a short delay since it's relying on the 1s clock resolution
        if self._total % 24 == 0:
            self._is_playing = self.info['time'] == self._last_time - 1
            self._last_time = self.info['time']
            if self._stoppage_detector.update(not self._is_playing):
                self._stoppage_count += 1

        self._stats["is_playing"] = self._is_playing
        self._stats["stoppage_count"] = self._stoppage_count
        self._stats["human_player"] = enum_to_str(self.info["human-player-enum"])

        if self._stats["is_playing"]:
            with self._filename.open("a+") as f:

                self.info.update(self._stats)
                f.write("{}\n".format(json.dumps(self.info)))

        return self._total

    @classmethod
    def fitness_threshold(cls) -> float:
        """
        Accessor fitness threshold (the score over
        which to stop training)
        """
        return 1e6  # Nothing relevant about this number
