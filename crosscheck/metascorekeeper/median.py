from typing import List
from ..scorekeeper import Scorekeeper
from .base import Metascorekeeper
import numpy as np


class Median(Metascorekeeper):

    def __init__(self):
        super().__init__()

    @property
    def score(self):
        return np.median(np.array([x.score for x in self._scorekeepers.values()]))

    @classmethod
    def fitness_threshold(cls, scorekeepers: List[Scorekeeper]) -> float:
        return sum([x.fitness_threshold() for x in scorekeepers]) / len(scorekeepers)

    def stats(self):
        stats = {}
        for sk in self._scorekeepers:
            stats.update(sk.stats)
        return stats

    def score_listing(self) -> List[float]:
        """
        Sorted for easier quick viewing of median
        """
        return sorted([x.score for x in self._scorekeepers.values()])
