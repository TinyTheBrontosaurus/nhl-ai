import abc
import numpy as np
from typing import List
from crosscheck.player.utils import RisingEdge
from PIL import Image, ImageDraw


class Display:
    @abc.abstractmethod
    def tick(self, buttons: dict):
        raise NotImplemented


class MenuHandler(Display):

    def __init__(self):
        self._stack = [TopMenu()]

    def tick(self, buttons: dict):
        if self.done:
           return

        active_menu = self._stack[0]
        active_menu.tick(buttons)
        next_menu = active_menu.next_menu

        if next_menu == active_menu:
            # Nothing to do
            pass
        elif next_menu is None:
            self._stack.pop(0)
        else:
            self._stack.insert(0, next_menu)

    @property
    def done(self):
        return len(self._stack) == 0

    @property
    def ob(self):
        if self.done:
            return None
        return self._stack[0].ob


class Menu(Display):

    def __init__(self):
        self.shape = (224, 320, 3)
        self._next_menu = self

    @property
    def ob(self) -> np.ndarray:
        ob = np.zeros(self.shape, dtype=np.uint8)

        blank_frame = np.zeros(ob.shape, dtype=np.uint8)
        img = Image.fromarray(blank_frame)
        draw = ImageDraw.Draw(img)

        for offset, line in enumerate(self.lines):
            draw.text((0, 5 + 12 * offset), line, fill='rgb(255, 255, 255)')

        return np.array(img)

    @property
    @abc.abstractmethod
    def lines(self) -> List[str]:
        raise NotImplemented

    @property
    def next_menu(self):
        return self._next_menu


class TopMenu(Menu):
    def __init__(self):
        super().__init__()
        self.selection = 0
        self.up = RisingEdge()
        self.down = RisingEdge()
        self.select = RisingEdge()
        self.back = RisingEdge()
        self._lines = []

    def tick(self, buttons: dict):

        self.up.update(buttons.get("UP"))
        self.down.update(buttons.get("DOWN"))
        self.select.update(buttons.get("A") or buttons.get("C") or buttons.get("START"))
        self.back.update(buttons.get("B"))

        options = ["Play minigame", "Quit"]

        if self.up.state:
            self.selection -= 1
            if self.selection < 0:
                self.selection = len(options) - 1
        if self.down.state:
            self.selection += 1
            if self.selection >= len(options):
                self.selection = 0

        self._lines = [f"{'> ' if idx == self.selection else '':2} {label}"
                       for idx, label in enumerate(options)]

        if self.select.state:
            if 1 == self.selection:
                self._next_menu = None
        if self.back.state:
            self._next_menu = None

    @property
    def lines(self):
        return self._lines
