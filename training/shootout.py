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
            [None, 'C'],
            *discretizers.IndependentDiscretizer.DPAD
         ])


class ShootoutTrainer(runner.Trainer):

    GOAL_Y = 256

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
    def get_config_filename(cls) -> str:
        """
        Accessor for config filename
        """
        return 'config-shootout'

    @classmethod
    def get_scenario_string(cls) -> str:
        """
        Accessor for the scenario to use
        """
        return 'ChiAtBuf-Shootout-Mogilny'

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

        puck_adjusted_goalie_x = min(max(puck_x, -23), 23)

        delta_puck_goalie_x = abs(puck_adjusted_goalie_x - goalie_x)

        # Lower numbers are better. but if behind the next (256) treat it the same as far from the net
        delta_puck_net_y = self.GOAL_Y - puck_y
        # If very close to the net (or behind), give the same credit as if above the slot
        if delta_puck_net_y <= 2:
            delta_puck_net_y = 100

        distance_multiplier = max(self.GOAL_Y - delta_puck_net_y, 0)

        juke_reward = delta_puck_goalie_x * distance_multiplier

        # Reward all jukes
        self._score_acc += juke_reward

        score = self._score_acc

        score += max(self._max_puck_y, self.GOAL_Y)
        score += max(self._max_shooter_y, self.GOAL_Y)

        if info['home-goals']:
            score += 100000

        # Stop early once stoppage occurs
        if info['shootout-stoppage']:
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

        self._stats = {"score": score}

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], ShootoutTrainer)
