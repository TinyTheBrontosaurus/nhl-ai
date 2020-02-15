from typing import List
from ..scorekeeper import Scorekeeper
from .base import Metascorekeeper
from .mean import Mean
from .median import Median

class NudgedMedian(Metascorekeeper):

    def __init__(self):
        super().__init__()
        self._mean = Mean()
        self._median = Median()

    def add(self, name: str, scorekeeper: Scorekeeper):
        """
        Add a scorekeeper
        :param name: Name of the keeper
        :param scorekeeper: The keeper
        """
        super().add(name, scorekeeper)
        self._mean.add(name, scorekeeper)
        self._median.add(name, scorekeeper)

    @property
    def score(self):
        lowest = min([x.score for x in self._scorekeepers.values()])
        return self._median.score + self._mean.score * 0.01 + lowest

    @classmethod
    def fitness_threshold(cls, scorekeepers: List[Scorekeeper]) -> float:
        return Median.fitness_threshold(scorekeepers) * 2.01

    def score_listing(self) -> List[float]:
        """
        Sorted for easier quick viewing of median
        """
        return self._median.score_listing()
