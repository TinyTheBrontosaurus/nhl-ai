from crosscheck.game_env import get_genv
from crosscheck.player.utils import RateController, RisingEdge
from crosscheck.player import human
from crosscheck import definitions
from loguru import logger
import retro
import gzip
import datetime
import pathlib
from gym.envs.classic_control.rendering import SimpleImageViewer
import numpy as np
from PIL import ImageDraw, Image
from crosscheck.version import __version__
from crosscheck.practice.menu import MenuHandler, Menu
from crosscheck.scorekeeper import Scorekeeper
from crosscheck.scorekeeper.shootout import Shootout
from typing import Optional


class RealTimeGame:

    def __init__(self, button_state: human.ButtonState, scenario: pathlib.Path,
                 viewer: SimpleImageViewer, menu: MenuHandler, scorekeeper_cls: Optional[Scorekeeper] = None,
                 timeout: Optional[float] = None, save_buttons=False):
        self.button_state = button_state
        self.scenario = scenario
        self._save_state_request = RisingEdge()
        self._done_request = RisingEdge()
        self.viewer = viewer
        self.frame = 0
        self.menu = menu
        self.scorekeeper_ctor = scorekeeper_cls
        self.scorekeeper = scorekeeper_cls
        self.timeout_frames = timeout * 60 if timeout is not None else None
        self.button_presses = []
        self.save_buttons = save_buttons
        from crosscheck.discretizers import Independent
        self.discretizer = Independent

    @classmethod
    def _save_state(cls, env):
        state_dir = definitions.NEW_SAVE_STATE_FOLDER
        label = datetime.datetime.now().isoformat().replace(":", "_")
        save_file = state_dir / f"{label}.state"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(save_file, 'wb') as f:
            f.write(env.em.get_state())
        logger.info(f"Saved state {str(save_file)}")

    def render(self, ob, *_args):

        self.menu.shape = ob.shape

        new_ob = ob
        if self.scorekeeper:
            blank_frame = np.zeros(ob.shape, dtype=np.uint8)
            img = Image.fromarray(blank_frame)
            draw = ImageDraw.Draw(img)

            sk = self.scorekeeper

            to_draw = dict()
            to_draw['version'] = __version__
            to_draw.update(sk.stats)

            score_vector = sk.score_vector
            total = sum(score_vector.values())
            score_breakdown = {"total": int(total)}

            for key, value in score_vector.items():
                try:
                    pct = value / total * 100.
                except ZeroDivisionError:
                    pct = 0
                score_breakdown[key] = "({pct:5,.1f}% {value:8,}".format(pct=pct, value=int(value))

            to_draw.update(score_breakdown)

            for offset, (key, value) in enumerate(to_draw.items()):
                draw.text((0, 5 + 12 * offset), "{:15}: {}".format(key, value), fill='rgb(255, 255, 255)')

            status_frame = np.array(img)

            new_ob = np.concatenate((ob, status_frame), axis=1)

        if False and not self.menu.done:
            new_ob = np.concatenate((ob, self.menu.ob), axis=1)

        self.viewer.imshow(new_ob)

    def play(self):

        env = get_genv()

        if self.discretizer is not None:
            env = self.discretizer(env, [[None, x] for x in ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']])

        env.use_restricted_actions = retro.Actions.ALL
        if self.scenario:
            env.load_state(str(self.scenario))
        env.reset()

        rate_controller = RateController(1/60)

        env.players = 1

        while not self.done:
            # Run the next step in the simulation
            with self.button_state.lock:
                next_action_dict = dict(self.button_state.state)

            # Save them all
            if self.save_buttons:
                self.button_presses.append(next_action_dict)
            #self.menu.tick(next_action_dict)

            # Convert to buttons
            next_action = [next_action_dict.get(key, False) for key in env.buttons]

            # Two player?
            #next_action.extend(next_action)

            step = env.step(next_action)

            info = step[3]
            if self.scorekeeper:
                self.scorekeeper.info = info
                self.scorekeeper.buttons_pressed = env.action_labels(next_action)
                self.scorekeeper.tick()

            self.render(*step)

            # Check custom button presses
            self._done_request.update(next_action_dict.get("X") or next_action_dict.get("Y"))
            self._save_state_request.update(next_action_dict.get("Z"))

            if self._save_state_request.state:
                self._save_state(env)

            rate_controller.tick()
            self.frame += 1

    @property
    def done(self) -> bool:
        return any([self._done_request.state,
                    self.menu.done,
                    self.frame > self.timeout_frames if self.timeout_frames is not None else False])
