import sys
from training import runner
from training import discretizers
import neat
import typing
from training.info_utils import get_player_w_puck, players_and_puck_feature


class ShootoutTrainer(runner.Trainer):

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, short_circuit: bool):
        super().__init__(genome, config, short_circuit)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        self._possession_frames = {
            'home': 0,
            None: 0,
            'away': 0
        }

        self._stats = {}

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

        self._possession_frames[player_w_puck.get('team')] += 1

        # 1 point for having the puck, 0 points for no one, -1 for opponent having the puck

        # 60 fps
        possession_score = self._possession_frames['home'] - self._possession_frames['away']

        # 10x for attack score vs possession
        attack_score = 600 * (info['home-attack-time'] - info['away-attack-time'])

        if info['time'] == 0:
            self._done = True

        score = possession_score + attack_score



        features = players_and_puck_feature(info)
        self._next_action = self.net.activate(features)

        self._stats = {"score": score,
                       "pos_home": self._possession_frames['home'],
                       "pos_none": self._possession_frames[None],
                       "pos_away": self._possession_frames['away'],
                       "atk_home": info['home-attack-time'],
                       "atk_away": info['away-attack-time']}

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], ShootoutTrainer)
