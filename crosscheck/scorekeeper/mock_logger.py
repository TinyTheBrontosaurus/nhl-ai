from .base import Scorekeeper
from crosscheck.log_folder import LogFolder
import json

class MockLogger(Scorekeeper):
    """
    This scorekeeper simply logs state
    """

    def __init__(self):
        super().__init__()
        self._total = 0
        self._filename = LogFolder.folder /"mocklogger.jsonl"

    def _tick(self):
        self._total += 1

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

    def stats(self) -> dict:
        return {}

    def score_vector(self) -> dict:
        return {}
