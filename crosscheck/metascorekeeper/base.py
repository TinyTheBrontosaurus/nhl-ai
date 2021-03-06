import abc
from typing import List, Dict
from ..scorekeeper import Scorekeeper

class Metascorekeeper:

    def __init__(self):
        self._scorekeepers: Dict[str, Scorekeeper] = {}

    @property
    @abc.abstractmethod
    def score(self) -> float:
        """
        Calculate the score from all scorekeepers
        :return:
        """
        pass

    @classmethod
    @abc.abstractclassmethod
    def fitness_threshold(cls, scorekeepers: List[Scorekeeper]) -> float:
        """
        Accessor fitness threshold (the score over
        which to stop training)
        """
        pass

    def add(self, name: str, scorekeeper: Scorekeeper):
        """
        Add a scorekeeper
        :param name: Name of the keeper
        :param scorekeeper: The keeper
        """
        self._scorekeepers[name] = scorekeeper

    @property
    def stats(self) -> dict:
        """
        Statistics on this object so far
        :return: key/value on statistics
        """
        return {key: value.stats() for key, value in self._scorekeepers.items()}

    def score_listing(self) -> List[float]:
        return [x.score for x in self._scorekeepers.values()]
