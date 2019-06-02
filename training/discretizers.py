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
    UP_DOWN = ["DOWN", None, "UP"]
    # LEFT and RIGHT can't be pressed at the same time
    LEFT_RIGHT = ["LEFT", None, "RIGHT"]
    DPAD = [LEFT_RIGHT, UP_DOWN]

    def __init__(self, env, independent_buttons: typing.List[typing.List[str]]):
        super(IndependentDiscretizer, self).__init__(env)
        # Values are:
        #  ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        self.buttons = env.buttons
        self._independent_buttons = independent_buttons
        self._all_false = (False,) * len(self.buttons)

    def action(self, a: typing.List[float]) -> list:
        if len(a) != len(self._independent_buttons):
            raise ValueError("Have {} outputs, expected {} outputs".format(len(a), len(self._independent_buttons)))
        buttons_pressed = list(self._all_false)

        for output, buttons in zip(a, self._independent_buttons):
            index = int(min(math.floor(output * len(buttons)), len(buttons) - 1))

            if buttons[index] is not None:
                button_string = buttons[index]
                pos = self.buttons.index(button_string)
                buttons_pressed[pos] = True

        return buttons_pressed


class Genesis3ButtonDiscretizer(IndependentDiscretizer):
    def __init__(self, env):
        super().__init__(env, [
            [None, 'A'],
            [None, 'B'],
            [None, 'C'],
            *IndependentDiscretizer.DPAD
         ])

class Genesis6ButtonDiscretizer(IndependentDiscretizer):
    def __init__(self, env):
        super().__init__(env, [
            [None, 'A'],
            [None, 'B'],
            [None, 'C'],
            [None, 'X'],
            [None, 'Y'],
            [None, 'Z'],
            *IndependentDiscretizer.DPAD
         ])
