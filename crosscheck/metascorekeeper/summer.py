from typing import List
from ..scorekeeper import Scorekeeper
from .base import Metascorekeeper


class Summer(Metascorekeeper):

    def __init__(self):
        super().__init__()

    @property
    def score(self):
        return sum([x.score for x in self._scorekeepers.values()])

    @classmethod
    def fitness_threshold(cls, scorekeepers: List[Scorekeeper]) -> float:
        return sum([x.fitness_threshold() for x in scorekeepers])

    def stats(self):
        stats = {}
        for sk in self._scorekeepers:
            stats.update(sk.stats)
        return stats