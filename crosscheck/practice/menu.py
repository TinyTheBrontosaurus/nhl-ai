import abc
import numpy as np
from typing import List
from crosscheck.player.utils import RisingEdge
from PIL import Image, ImageDraw
import yaml
import crosscheck.definitions


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
            if not self.done:
                # Reset any button monitors from last time
                self._stack[0].enter()
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

    def enter(self):
        pass


class BasicMenu(Menu):
    def __init__(self, options: dict):
        super().__init__()
        self.selection = 0
        self.up = RisingEdge()
        self.down = RisingEdge()
        self.select = RisingEdge()
        self.back = RisingEdge()
        self._lines = []
        self.options = options

    def tick(self, buttons: dict):

        self.up.update(buttons.get("UP"))
        self.down.update(buttons.get("DOWN"))
        self.select.update(buttons.get("A") or buttons.get("C") or buttons.get("START"))
        self.back.update(buttons.get("B"))


        if self.up.state:
            self.selection -= 1
            if self.selection < 0:
                self.selection = len(self.options) - 1
        if self.down.state:
            self.selection += 1
            if self.selection >= len(self.options):
                self.selection = 0

        self._lines = [f"{'> ' if idx == self.selection else '':2} {label}"
                       for idx, label in enumerate(self.options.keys())]

        # Default to no-op
        self._next_menu = self

        # If select was chosen, go to next menu
        if self.select.state:
            next_menus = list(self.options.values())
            if self.selection >= len(next_menus):
                self._next_menu = None
            else:
                next_menu_type = next_menus[self.selection]
                if next_menu_type is None:
                    self._next_menu = next_menu_type
                else:
                    self._next_menu = next_menu_type()
        if self.back.state:
            self._next_menu = None

    @property
    def lines(self):
        return self._lines

    def __call__(self):
        return self

    def enter(self):
        self.up.reset()
        self.down.reset()
        self.select.reset()
        self.back.reset()


class TopMenu(BasicMenu):
    def __init__(self):
        super().__init__({
            "Play minigame": MinigameMenu,
            "Quit": None
        })


class MinigameMenu(BasicMenu):

    def __init__(self):
        folder = crosscheck.definitions.SAVE_STATE_FOLDER / "02-2p"
        manifest = folder / "manifest.yml"

        with manifest.open() as f:
            self._manifest = yaml.safe_load(f)

        options = {f"{x['name']} ({x['file']})": self for x in self._manifest}
        super().__init__(options)

