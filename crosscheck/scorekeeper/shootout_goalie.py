import typing
from .base import Scorekeeper
from ..info_utils.wrapper import InfoAccumulator, InfoWrapper

class ShootoutGoalie(Scorekeeper):

    def __init__(self):
        super().__init__()

        self._accumulator = InfoAccumulator()

        self._pressed = {x: 0 for x in ['A', 'B', 'C']}

        self._juke_accumulator = 0

        self._shot_score = 0

    def _tick(self) -> float:
        """
        Incorporate the latest frame into the total score
        :return: The total score as of this frame
        """

        # Update stats
        self._accumulator.info = self.info
        self._accumulator.accumulate()

        # End when play stops. shootout itself ends after 25s
        self._done_reasons['shoot_stopped'] = self.info['shootout-stoppage'] > 0


        self._score_vector = {}
        score_vector = self._score_vector
        score_vector['away-goals'] = (1-self.info['away-goals']) * 1e4

        score = sum(score_vector.values())

        # Accumulate button presses
        if 'A' in self.buttons_pressed:
            self._pressed['A'] += 1
        if 'B' in self.buttons_pressed:
            self._pressed['B'] += 1
        if 'C' in self.buttons_pressed:
            self._pressed['C'] += 1

        # Save the score vector
        self._score_vector = score_vector

        # Stats to track
        self._stats = {
            'shootout-stopped': "{}".format(self.info['shootout-stoppage']),
            'buttons': self._pressed,
        }

        return score


    @classmethod
    def fitness_threshold(cls) -> float:
        """
        Accessor fitness threshold (the score over
        which to stop training)
        """
        return 1e4
