import gym
import numpy as np
import math
import sys
from training import runner
import neat
import typing


class Discretizer(gym.ActionWrapper):
    """
    Simplify inputs to win a faceoff
    """
    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        # Values are:
        #  ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        buttons = env.buttons

        # Create all combinations on the D pad for pressing B or not (2*3*3==18)
        actions = []
        for button in ['B', None]:
            for lr in ['LEFT', 'RIGHT', None]:
                for ud in ['UP', 'DOWN', None]:
                    action = []
                    if button:
                        action.append(button)
                    if lr:
                        action.append(lr)
                    if ud:
                        action.append(ud)
                    actions.append(action)

        # Convert easy-to-debug strings to boolean array
        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a: typing.List[float]) -> list:
        index = math.floor(a[0] * len(self._actions))
        if index >= len(self._actions):
            index = len(self._actions) - 1
        return self._actions[index].copy()


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
    def discretizer_class(cls) -> typing.Callable[[], Discretizer]:
        return Discretizer

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
    runner.main(sys.argv[1:], FaceoffTrainer)
