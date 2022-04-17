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
from typing import Optional, Callable
from crosscheck.player.utils import RisingEdge
from crosscheck.player import ai as player_ai


class RealTimeGame:

    def __init__(self, button_state: human.ButtonState, scenario: Optional[pathlib.Path],
                 viewer: SimpleImageViewer, menu: MenuHandler,
                 scorekeeper_cls: Optional[Callable[[], Scorekeeper]] = None,
                 timeout: Optional[float] = None, save_buttons=False,
                 exit_on_y=False):
        self.button_state = button_state
        self.scenario = scenario
        self._save_state_request = RisingEdge()
        self._done_request = RisingEdge()
        self.viewer = viewer
        self.frame = 0
        self.menu = menu
        self.scorekeeper_ctor = scorekeeper_cls
        self.scorekeeper = self.scorekeeper_ctor()
        self.timeout_frames = timeout * 60 if timeout is not None else None
        self.button_presses = []
        self.save_buttons = save_buttons
        self.discretizer = None
        self.exit_on_y = exit_on_y

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
        """
        Loop to play the game nominally. Uses a physical controller as 1p and optionally an AI as 2p
        """

        env = get_genv()

        if self.discretizer is not None:
            env = self.discretizer(env, [[None, x] for x in ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']])

        # Load the start state, if there is one
        env.use_restricted_actions = retro.Actions.ALL
        if self.scenario:
            env.load_state(str(self.scenario))
        env.reset()
        env.players = 2

        # Maintain framerate since there is a human playing
        rate_controller = RateController(1/60)

        # Keep track of IO for enabling/disabling AI for 2p.
        press_2p = RisingEdge()
        enabled_2p = False
        ai_player = player_ai.AiPlayer(env)
        next_action_2p_disabled = [False for _ in range(env.num_buttons)]

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

            # Two player
            if enabled_2p:
                next_action_2p = next_action + ai_player.next_action
            else:
                next_action_2p = next_action + next_action_2p_disabled

            step = env.step(next_action_2p)

            info = step[3]

            # AI's next step
            ai_player.step(info)

            if self.scorekeeper:
                self.scorekeeper.info = info
                self.scorekeeper.buttons_pressed = next_action_dict
                self.scorekeeper.tick()

            self.render(*step)

            # Check custom button presses
            press_2p.update(next_action_dict.get("Y"))
            if press_2p.state:
                enabled_2p = not enabled_2p
                logger.info("2P AI {value}abled", value="en" if enabled_2p else "dis")
            self._done_request.update(next_action_dict.get("X") or (self.exit_on_y and next_action_dict.get("Y")))
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
