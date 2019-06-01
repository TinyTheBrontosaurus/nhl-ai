import sys
from training import runner
from training import discretizers
import neat
import typing
from training.info_utils import get_player_w_puck


class BButtonDiscretizer(discretizers.IndependentDiscretizer):
    """
    For a faceoff, only the "B" button and DPAD is important
    """
    def __init__(self, env):
        super().__init__(env, [
            ['B', None],
            *discretizers.IndependentDiscretizer.DPAD
         ])


class FaceoffTrainer(runner.Trainer):
    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, short_circuit: bool):
        super().__init__(genome, config, short_circuit)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        self._frame = 0
        self._score = 0
        self._done = False
        self._min_puck_y = 500
        self._puck_y = 0
        self._faceoff_winner = None
        self._stats = {}
        # No bonus until faceoff is won
        self._puck_bonus = 0

        # The first action
        features = [self._frame]
        self._next_action = self.net.activate(features)

    @classmethod
    def discretizer_class(cls) -> typing.Callable[[], BButtonDiscretizer]:
        return BButtonDiscretizer

    @property
    def next_action(self) -> list:
        return self._next_action

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def done(self) -> bool:
        return self._done

    def tick(self, ob, rew, done, info) -> float:
        faceoffs_won = info['home-faceoff']
        faceoffs_lost = info['away-faceoff']
        puck_y = info['puck-ice-y']
        total_faceoffs = faceoffs_won + faceoffs_lost
        self._min_puck_y = min(puck_y, self._min_puck_y)

        # First to touch the puck (not None and not {}) wins the faceoff
        if total_faceoffs > 0 and not self._faceoff_winner and faceoffs_won > 0:
            self._faceoff_winner = get_player_w_puck(info)

            # Incentivize RD to win
            bonuses = {
                'LW': 200,
                'C': 300,
                'RW': 500,
                'LD': 100,
                'RD': 1000,
                'G': 20
            }

            if self._faceoff_winner.get('team') == 'home':
                self._puck_bonus = bonuses.get(self._faceoff_winner.get('pos'), 0)

        # Always end after this time
        if self._frame > 300:
            self._done = True

        # Stop early once a faceoff has been won and someone has the puck
        if self.short_circuit and self._puck_bonus > 0 and faceoffs_won > 0:
            self._done = True

        score = (faceoffs_won - faceoffs_lost) * 100 + -self._min_puck_y + self._puck_bonus

        # Calculate action for the next frame
        self._frame += 1
        features = [self._frame]
        self._next_action = self.net.activate(features)

        self._stats = {"score": score, "frame": self._frame, "puck_y": puck_y, "faceoff_winner": self._faceoff_winner}

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], FaceoffTrainer)
