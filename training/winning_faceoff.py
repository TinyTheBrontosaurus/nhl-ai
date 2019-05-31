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
        self._player_w_puck = None
        self._stats = {}

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

        player_w_puck = get_player_w_puck(info)

        # Incentivize RD to win
        bonuses = {
            'LW': 200,
            'C': 300,
            'RW': 500,
            'LD': 100,
            'RD': 1000,
            'G': 20
        }

        puck_bonus = 0
        if player_w_puck.get('team') == 'home':
            puck_bonus = bonuses.get(player_w_puck.get('pos'), 0)

        # No bonus before a faceoff is won
        if faceoffs_won <= 0:
            puck_bonus = 0

        # Always end after this time
        if self._frame > 300:
            self._done = True

        # Stop early once a faceoff has been won and someone has the puck
        if self.short_circuit and puck_bonus > 0 and faceoffs_won > 0:
            self._done = True

        score = (faceoffs_won - faceoffs_lost) * 100 + -self._min_puck_y + puck_bonus

        # Calculate action for the next frame
        self._frame += 1
        features = [self._frame]
        self._next_action = self.net.activate(features)

        self._stats = {"score": score, "frame": self._frame, "puck_y": puck_y, "player_w_puck": player_w_puck}

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], FaceoffTrainer)
