import sys
from training import runner, discretizers
import neat
import typing


class AggroTrainer(runner.Trainer):
    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, short_circuit: bool):
        super().__init__(genome, config, short_circuit)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        self._score = 0
        self._done = False
        self._stats = {}

        # The first action
        self._next_action = [0] * 12

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
        home_checks = info['home-checks']
        away_checks = info['away-checks']
        puck_y = info['puck-ice-y']
        total_faceoffs = faceoffs_won + faceoffs_lost
        self._min_puck_y = min(puck_y, self._min_puck_y)

        if info['player-w-puck-ice-x'] == info['player-home-7-x'] and \
                        info['player-w-puck-ice-y'] == info['player-home-7-y']:
            player_w_puck = 7
            puck_bonus = 1000
        elif info['player-w-puck-ice-x'] == info['player-home-10-x'] and \
                        info['player-w-puck-ice-y'] == info['player-home-10-y']:
            player_w_puck = 10
            puck_bonus = 200
        elif info['player-w-puck-ice-x'] == info['player-home-16-x'] and \
                        info['player-w-puck-ice-y'] == info['player-home-16-y']:
            player_w_puck = 16
            puck_bonus = 300
        elif info['player-w-puck-ice-x'] == info['player-home-89-x'] and \
                        info['player-w-puck-ice-y'] == info['player-home-89-y']:
            player_w_puck = 89
            puck_bonus = 500
        elif info['player-w-puck-ice-x'] == info['player-home-8-x'] and \
                        info['player-w-puck-ice-y'] == info['player-home-8-y']:
            player_w_puck = 8
            puck_bonus = 100
        elif info['player-w-puck-ice-x'] == info['player-home-goalie-ice-x'] and \
                        info['player-w-puck-ice-y'] == info['player-home-goalie-ice-y']:
            player_w_puck = 31
            puck_bonus = 20
        else:
            player_w_puck = None
            puck_bonus = 0

        if faceoffs_won <= 0:
            puck_bonus = 0

        if self._frame > 300:
            self._done = True

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
    runner.main(sys.argv[1:], AggroTrainer)
