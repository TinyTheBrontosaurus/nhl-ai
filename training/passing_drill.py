import sys
from training import runner
from training import discretizers
import neat
import typing
from training.info_utils import InfoAccumulator
import math


class PassingDrillTrainer(runner.Trainer):

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, short_circuit: bool):
        super().__init__(genome, config, short_circuit)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        self._stats = {}

        self._done = False

        self._next_action = None

        self._accumulator = InfoAccumulator()
        self._b_pressed = 0

    @classmethod
    def get_config_filename(cls) -> str:
        """
        Accessor for config filename
        """
        return 'config-ballhog'

    @classmethod
    def get_scenario_string(cls) -> str:
        """
        Accessor for the scenario to use
        """
        return 'ChiAtBuf-Faceoff'

    @classmethod
    def discretizer_class(cls) -> typing.Callable[[], discretizers.Genesis3ButtonDiscretizer]:
        return discretizers.Genesis3ButtonDiscretizer

    @property
    def next_action(self) -> list:
        return self._next_action

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def done(self) -> bool:
        return self._done

    def tick(self, ob, rew, done, info, env) -> float:

        # Update stats
        self._accumulator.info = info
        self._accumulator.accumulate()

        # End when the other team gets the puck, home own goals, or time runs out
        att = self._accumulator.pass_attempts['home']
        cmp = self._accumulator.pass_completions['home']

        cmp_pct = 0.0

        if att > 1:
            cmp_pct = cmp / (att - 1)  # The latest one can't be counted since it might still be happening
            threshold = 0.75  # 3/4 must be completed

            if cmp_pct < threshold:
                self._done = True
        elif self._accumulator.wrapper.player_w_puck.get('team') == 'away':
            self._done = True

        if info['time'] == 0 or info['away-goals'] > 0:
            self._done = True

        # Rewards
        score_vector = []
        score_vector.append(self._accumulator.time_puck['home'] * 2)
        score_vector.append(self._accumulator.time_puck[None] * 1)
        score_vector.append(self._accumulator.time_puck['away'] * -3)
        score_vector.append(self._accumulator.pass_attempts['home'] * 3)
        score_vector.append(self._accumulator.pass_completions['home'] * 10)
        consecutive_pass_vector = [math.pow(2, x) for x in self._accumulator.consecutive_passes['consecutive']['home']]
        score_vector.append(sum(consecutive_pass_vector))
        score = sum(score_vector)

        # Calculate commands based on features
        self._next_action = self.net.activate(self._accumulator.wrapper.players_and_puck_feature)
        buttons_pressed = env.action_labels(self._next_action)
        if 'B' in buttons_pressed:
            self._b_pressed += 1

        # Stats to track
        self._stats = {
            'score': score,
            'time_w_puck': self._accumulator.time_puck['home'],
            'time_no_puck': self._accumulator.time_puck[None],
            'time_opp_puck': self._accumulator.time_puck['away'],
            'successful_passes': self._accumulator.pass_completions,
            'consecutive_passes': self._accumulator.consecutive_passes['consecutive']['home'],
            'cmp_pct': cmp_pct,
            'unique_passes': self._accumulator.consecutive_passes['unique']['home'],
            'b_pressed': self._b_pressed,
        }

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], PassingDrillTrainer)
