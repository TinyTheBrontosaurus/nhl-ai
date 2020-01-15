from .base import Combiner

class Summer(Combiner):

    def __init__(self):
        super().__init__()

    @property
    def score(self):
        return sum([x.score for x in self._scorekeepers])
