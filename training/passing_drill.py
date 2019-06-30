import sys
from training import runner
from training import discretizers
import neat
import typing
from training.info_utils import get_player_w_puck, players_and_puck_feature


class PassingDrillTrainer(runner.Trainer):

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, short_circuit: bool):
        super().__init__(genome, config, short_circuit)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        self._last_player_w_puck = None

        self._pass_accumulator = 0

        self._stats = {
            'score': 0,
            'time_w_puck': 0.0,
            'time_wo_puck': 0.0,
            'successful_passes': 0,
            'consecutive_passes': 0,
            'unique_passes': 0,
            'consecutive_unique_passes': 0,
        }

        self._recent_positions = []

        self._done = False

        self._next_action = None

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

    def tick(self, ob, rew, done, info) -> float:

        player_w_puck = get_player_w_puck(info)

        pass_score_this_frame = 0

        if player_w_puck.get('team') == 'home':
            self._stats['time_w_puck'] += 1.0 / 60
            last_pos = self._last_player_w_puck.get('pos')
            pos = player_w_puck.get('pos')

            # No points for turnover, but points for passing
            if last_pos is not None and last_pos != pos:
                # Successful pass
                self._stats['successful_passes'] += 1
                self._stats['consecutive_passes'] += 1

                pass_score_this_frame += self._stats['consecutive_passes'] * 100

                # Pass to new player?
                if pos not in self._recent_positions:
                    self._stats['unique_passes'] += 1
                    self._stats['consecutive_unique_passes'] += 1
                    self._recent_positions.append(pos)

                    pass_score_this_frame += self._stats['consecutive_unique_passes'] * 1000

                    # Always have it so at least two players can get passed the puck
                    if len(self._recent_positions) == 5:
                        self._recent_positions.pop(0)
                else:
                    self._stats['consecutive_unique_passes'] = 0



        elif player_w_puck.get('team') == 'away':
            self._last_player_w_puck = None
            self._stats['time_wo_puck'] += 1.0 / 60
            self._stats['consecutive_passes'] = 0
            self._stats['consecutive_unique_passes'] = 0

        self._pass_accumulator += pass_score_this_frame

        score = self._pass_accumulator + self._stats['time_w_puck']

        if info['time'] == 0:
            self._done = True

        features = players_and_puck_feature(info)
        self._next_action = self.net.activate(features)

        self._stats['score'] = score

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], PassingDrillTrainer)
