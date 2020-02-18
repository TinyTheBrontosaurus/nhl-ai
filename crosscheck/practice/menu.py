import abc
import numpy as np
from typing import List
from crosscheck.player.utils import RisingEdge
from PIL import Image, ImageDraw


class Display:
    @abc.abstractmethod
    def tick(self, buttons: dict):
        raise NotImplemented

class Menu(Display):

    def __init__(self):
        self.shape = (240, 180)

    def tick(self, buttons: dict):
        self._update(buttons)

    @property
    def ob(self) -> np.ndarray:
        ob = np.zeros(self.shape, dtype=np.uint8)

        blank_frame = np.zeros(ob.shape, dtype=np.uint8)
        img = Image.fromarray(blank_frame)
        draw = ImageDraw.Draw(img)

        for offset, line in enumerate(self.lines):
            draw.text((0, 5 + 12 * offset), line, fill='rgb(255, 255, 255)')

        return np.array(img)

    @abc.abstractmethod
    def _update(self, buttons: dict):
        raise NotImplemented

    @property
    @abc.abstractmethod
    def lines(self) -> List[str]:
        raise NotImplemented


class TopMenu(Menu):
    def __init__(self):
        super().__init__()
        self.selection = 0
        self.up = RisingEdge()
        self.down = RisingEdge()
        self.select = RisingEdge()
        self._lines = []

    def _update(self, buttons: dict):

        self.up.update(buttons.get("UP"))
        self.down.update(buttons.get("DOWN"))
        self.select.update(buttons.get("A") or buttons.get("B") or buttons.get("C") or buttons.get("START"))

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

    @property
    def lines(self):
        return self._lines
