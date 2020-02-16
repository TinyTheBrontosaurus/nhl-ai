import gym
import typing
import math
import abc


class Independent(gym.ActionWrapper):
    """
    Simplify inputs on the D pad. Otherwise 12 inputs
    """

    # UP and DOWN can't be pressed at the same time
    UP_DOWN = ["DOWN", None, "UP"]
    # LEFT and RIGHT can't be pressed at the same time
    LEFT_RIGHT = ["LEFT", None, "RIGHT"]
    DPAD = [LEFT_RIGHT, UP_DOWN]

    def __init__(self, env, independent_buttons: typing.List[typing.List[str]]):
        super().__init__(env)
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

    def action_labels(self, a: typing.List[float]):
        return [button for button, pressed in zip(self.buttons, self.action(a)) if pressed]

    @classmethod
    def button_count(cls) -> int:
        # Hardcoded due to chicken/egg
        return 10


class Genesis3Button(Independent):
    """
    Only control 3 button (A, B, C) plus the D pad
    """

    buttons_options = [
            [None, 'A'],
            [None, 'B'],
            [None, 'C'],
            *Independent.DPAD
         ]
    def __init__(self, env):
        super().__init__(env, self.buttons_options)

    @classmethod
    def button_count(cls) -> int:
        return len(cls.buttons_options)


class Genesis6Button(Independent):
    """
    Only control 6 button (A, B, C, X, Y, Z) plus the D pad
    """

    buttons_options = [
            [None, 'A'],
            [None, 'B'],
            [None, 'C'],
            [None, 'X'],
            [None, 'Y'],
            [None, 'Z'],
            *Independent.DPAD
         ]

    def __init__(self, env):
        super().__init__(env, self.buttons_options)

    @classmethod
    def button_count(cls) -> int:
        return len(cls.buttons_options)


class Genesis2ButtonBc(Independent):
    """
    Only control 2 buttons (B, C) plus the D pad
    """

    buttons_options = [
            [None, 'B'],
            [None, 'C'],
            *Independent.DPAD
         ]
    def __init__(self, env):
        super().__init__(env, self.buttons_options)

    @classmethod
    def button_count(cls) -> int:
        return len(cls.buttons_options)


string_to_class = {
    '3-button': Genesis3Button,
    '6-button': Genesis6Button,
    '8-button': Independent,
    '2-button-bc': Genesis2ButtonBc,
}
