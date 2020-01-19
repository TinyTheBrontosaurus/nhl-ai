import abc
from ..scorekeeper import Scorekeeper

class Metascorekeeper:

    def __init__(self):
        self._scorekeepers = {}

    @property
    @abc.abstractmethod
    def score(self) -> float:
        pass

    def add(self, name: str, scorekeeper: Scorekeeper):
        self._scorekeepers[name] = scorekeeper

    @property
    @abc.abstractmethod
    def stats(self) -> dict:
        pass

