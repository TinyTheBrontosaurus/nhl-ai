import abc

class Combiner:

    def __init__(self):
        self._scorekeepers = {}

    @abc.abstractmethod
    @property
    def score(self):
        pass

    def add(self, scorekeeper):
        pass
