import typing
from .base import Scorekeeper
from ..info_utils.wrapper import InfoAccumulator, InfoWrapper

class ScoreOnly(Scorekeeper):

    def __init__(self):
        super().__init__()

    def _tick(self) -> float:
        """
        Incorporate the latest frame into the total score
        :return: The total score as of this frame
        """

        self._score_vector = {'home-goals': self.info['home-goals']}

        score = sum(self._score_vector.values())

        # Stats to track
        self._stats = {}

        return score

    @classmethod
    def fitness_threshold(cls) -> float:
        """
        Accessor fitness threshold (the score over
        which to stop training)
        """
        return 1
