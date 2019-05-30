import gym
import typing
import math
import numpy as np
import itertools


class IndependentDiscretizer(gym.ActionWrapper):
    """
    Simplify inputs
    """

    # UP and DOWN can't be pressed at the same time
    UP_DOWN = ["UP", "DOWN", None]
    # LEFT and RIGHT can't be pressed at the same time
    LEFT_RIGHT = ["LEFT", "RIGHT", None]
    DPAD = [LEFT_RIGHT, UP_DOWN]

    def __init__(self, env, independent_buttons: typing.List[typing.List[str]]):
        super(IndependentDiscretizer, self).__init__(env)
        # Values are:
        #  ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        buttons = env.buttons

        # Create all combinations on the D pad for pressing B or not (2*3*3==18)
        actions = []
        for button_combo in itertools.product(independent_buttons):
            none_less_combo = [x for x in button_combo if x is not None]
            actions.append(none_less_combo)

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


class Genesis3ButtonDiscretizer(IndependentDiscretizer):
    def __init__(self, env):
        super().__init__(env, [
            ['A', None],
            ['B', None],
            ['C', None],
            *IndependentDiscretizer.DPAD
         ])

class Genesis6ButtonDiscretizer(IndependentDiscretizer):
    def __init__(self, env):
        super().__init__(env, [
            ['A', None],
            ['B', None],
            ['C', None],
            ['X', None],
            ['Y', None],
            ['Z', None],
            *IndependentDiscretizer.DPAD
         ])
