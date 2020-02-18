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


class RealTimeGame:

    def __init__(self, button_state: human.ButtonState, scenario: pathlib.Path,
                 viewer: SimpleImageViewer, menu: MenuHandler):
        self.button_state = button_state
        self.scenario = scenario
        self._save_state_request = RisingEdge()
        self._done_request = RisingEdge()
        self.viewer = viewer
        self.frame = 0
        self.menu = menu

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

        if not self.menu.done:
            new_ob = np.concatenate((ob, self.menu.ob), axis=1)
            self.viewer.imshow(new_ob)

    def play(self):

        env = get_genv()
        env.use_restricted_actions = retro.Actions.ALL
        if self.scenario:
            env.load_state(str(self.scenario))
        env.reset()

        rate_controller = RateController(1/60)

        env.players = 1

        while not self._done_request.state and not self.menu.done:
            # Run the next step in the simulation
            with self.button_state.lock:
                next_action_dict = dict(self.button_state.state)

            self.menu.tick(next_action_dict)

            # Convert to buttons
            next_action = [next_action_dict.get(key, 0) > 0.5 for key in env.buttons]

            # Two player?
            #next_action.extend(next_action)

            _step = env.step(next_action)

            self.render(*_step)

            # Check custom button presses
            self._done_request.update(next_action_dict.get("X") > 0.5)
            self._save_state_request.update(next_action_dict.get("Z") > 0.5)

            if self._save_state_request.state:
                self._save_state(env)

            rate_controller.tick()
            self.frame += 1
