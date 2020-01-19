from .base import Metascorekeeper

class Summer(Metascorekeeper):

    def __init__(self):
        super().__init__()

    @property
    def score(self):
        return sum([x.score for x in self._scorekeepers.values()])
