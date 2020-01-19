import abc

class Metascorekeeper:

    def __init__(self):
        self._scorekeepers = {}

    @property
    @abc.abstractmethod
    def score(self):
        pass

    def add(self, scorekeeper):
        pass
