import sys
from training import runner
from training import discretizers
import neat
import typing
from training.info_utils import get_player_w_puck


class CButtonDiscretizer(discretizers.IndependentDiscretizer):
    """
    For a shootouts, only the "C" button and DPAD is important
    """
    def __init__(self, env):
        super().__init__(env, [
            ['C', None],
            *discretizers.IndependentDiscretizer.DPAD
         ])


class ShootoutTrainer(runner.Trainer):
    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, short_circuit: bool):
        super().__init__(genome, config, short_circuit)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        self._score_acc = 0
        self._stats = {}
        self._max_puck_y = -500
        self._max_shooter_y = -500

        self._done = False

        self._next_action = None

    @classmethod
    def discretizer_class(cls) -> typing.Callable[[], CButtonDiscretizer]:
        return CButtonDiscretizer

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
        # The first action
        # inputs: player x, y, puck x, y, goalie x, y
        # outputs: ud, lr, c


        shooter_x = info['player-home-shootout-x']
        shooter_y = info['player-home-shootout-y']
        goalie_x = info['player-away-goalie-x-ice']
        goalie_y = info['player-away-goalie-y-ice']
        puck_x = info['puck-ice-x']
        puck_y = info['puck-ice-y']

        self._max_puck_y = max(puck_y, self._max_puck_y)
        self._max_shooter_y = max(puck_y, self._max_shooter_y)

        # goalie x bounds: (-23, 23)
        # goalie y bounds: (245, 256)
        # end line y: ~256
        # puck in net: 272
        #
        # reward:
        # * difference between puck and goalie, bounded. X preferred. bonus as Y is different
        # * x speed of puck?
        # * y position of player, bounded. encourage moving towards goalie
        # * shot location? want to discourage wasted shot
        # * time elapsed? again, want to dicourage wasted shot

        puck_adjusted_goalie_x = min(max(puck_x, 245), 256)

        delta_puck_goalie_x = abs(puck_adjusted_goalie_x - goalie_x)

        # Lower numbers are better. but if behind the next (256) treat it the same as far from the net
        delta_puck_net_y = 100 if puck_y > 255 else 256 - puck_y



        score = self._score_acc
        if info['home-goals']:
            score += 10000

        # Stop early once stoppage occurs
        if self.short_circuit and info['shootout_stoppage']:
            self._done = True

        features = [
            shooter_x,
            shooter_y,
            goalie_x,
            goalie_y,
            puck_x,
            puck_y
        ]
        self._next_action = self.net.activate(features)

        self._stats = {"score": score, "frame": self._frame, "puck_y": puck_y, "faceoff_winner": self._faceoff_winner}

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], ShootoutTrainer)
