import argparse
import sys
from crosscheck.game_env import get_genv
from crosscheck.player import human
from .config import cc_config
import threading
import time
from loguru import logger
import retro
import gzip
import crosscheck.definitions
import crosscheck.config
import datetime
import confuse
import pathlib


template = {
    # The name of this configuration (for logging purposes)
    'name': str,
    # The mode in which to run
    #  'train': Create a new model
    #  'replay': Run an existing model
    #  'compete': Run two models against each other
    #  'play': Run a model against a human player (controller input)
    #  'play-2p': Play traditional human vs human
    'mode': confuse.OneOf(['train', 'replay', 'compete', 'play', 'play-2p']),
    # The 1+ scenarios that are run in serial
    'scenarios': confuse.Sequence({
        # The filename of a scenario (save state) from which to start play
        'save-state': str,
    }),
}


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 practice scenarios')
    parser.add_argument('--config', '-c', help='App config.')

    args = parser.parse_args(argv)

    # Parse the config using confuse, bailing on failure
    try:

        if args.config:
            cc_config.set_file(args.config)
        cc_config.set_args(args, dots=True)

        logger.debug('configuration directory is {}', cc_config.config_dir())

        valid_config = cc_config.get(template)

        crosscheck.config.filename = args.config

    except confuse.ConfigError as ex:
        logger.critical("Problem parsing config: {}", ex)
        return

    scenarios = [load_save_state(x['save-state'].get()) for x in cc_config['scenarios']]

    button_state = human.ButtonState()
    button_thread = threading.Thread(target=human.maintain_button_state, args=(button_state,))
    button_thread.start()

    try:
        player = Player(scenarios[0], button_state)
        player.play()
    finally:
        button_state.running = False



def load_save_state(name: str) -> pathlib.Path:
    """
    Load the scenario, and verify the file exists
    :param name: The name of a scenario
    :return: The fully qualified path to the scenario
    """
    filename = definitions.SAVE_STATE_FOLDER / name
    if not filename.is_file():
        raise FileNotFoundError("Cannot find scenario {}".format(filename))
    return filename


class RateController:

    def __init__(self, time_per_iter):
        self.time_per_iter = time_per_iter
        self._next_time = self.reinit()

    def reinit(self):
        self._next_time = time.time() + self.time_per_iter
        return self._next_time

    def tick(self):
        now = time.time()
        delay_needed = self._next_time - now
        if delay_needed > 0:
            time.sleep(delay_needed)
        else:
            logger.warning(f"Falling behind {-delay_needed:.3f}s")
            next_time = now
        self._next_time += self._time_per_frame


class RisingEdge:
    def __init__(self):
        self._request_high = True
        self.state = False

    def update(self, request):
        # Return True only on a rising edge
        self.state = False

        if not self._request_high and request:
            self.state = True
            self._request_high = True
        self._request_high = request

        return self.state

class Player:

    def __init__(self, button_state: human.ButtonState, scenario: pathlib.Path):
        self.button_state = button_state
        self.scenario = scenario
        self._save_state_request = RisingEdge()
        self._done_request = RisingEdge()

    @classmethod
    def _save_state(cls, env):
        state_dir = crosscheck.definitions.NEW_SAVE_STATE_FOLDER
        label = datetime.datetime.now().isoformat().replace(":", "_")
        save_file = state_dir / f"{label}.state"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(save_file, 'wb') as f:
            f.write(env.em.get_state())
        logger.info(f"Saved state {str(save_file)}")

    def play(self):

        env = get_genv()
        env.use_restricted_actions = retro.Actions.ALL
        env.initial_state = self.scenario
        env.reset()

        rate_controller = RateController(1/60)

        env.players = 1
        frame = 0

        while not self._done_request.state:
            # Run the next step in the simulation
            with self.button_state.lock:
                next_action_dict = dict(button_state.state)

            # Convert to buttons
            next_action = [next_action_dict.get(key, 0) > 0.5 for key in env.buttons]

            # Two player?
            #next_action.extend(next_action)

            _step = env.step(next_action)

            env.render()

            # Check custom button presses
            self._done_request.update(next_action_dict.get("X") > 0.5)
            self._save_state_request.update(next_action_dict.get("Z") > 0.5)

            if self._save_state_request.state:
                self._save_state(env)

            rate_controller.tick()
            frame += 1


if __name__ == "__main__":
    main(sys.argv[1:])
