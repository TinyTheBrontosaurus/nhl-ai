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
        self._frame = 0

        # The first action
        features = [self._frame]
        self._next_action = self.net.activate(features)

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

        # Play for a minute
        if info['time'] < 600 - 60:
            self._done = True

        # Pure score on checks
        score = (home_checks * 100) - (away_checks * 50)

        # Calculate action for the next frame
        self._frame += 1
        features = [self._frame]
        self._next_action = self.net.activate(features)

        self._stats = {"score": score}

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], AggroTrainer)
