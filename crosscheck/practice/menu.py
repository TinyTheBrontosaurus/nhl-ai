import abc
import numpy as np
from typing import List
from crosscheck.player.utils import RisingEdge


class Display:
    @abc.abstractmethod
    def tick(self, buttons: dict):
        raise NotImplemented

class Menu(Display):

    @abc.abstractmethod
    @property
    def lines(self) -> List[str]:
        raise NotImplemented


class TopMenu(Display):
    def __init__(self):
        self.selection = 0
        self.up = RisingEdge()
        self.down = RisingEdge()
        self.select = RisingEdge()
        self._lines = []

    def tick(self, buttons: dict):

        self.up.update(buttons.get("UP"))
        self.down.update(buttons.get("UP"))
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
