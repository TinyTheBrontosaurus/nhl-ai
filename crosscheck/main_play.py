import argparse
import sys
from crosscheck.player import human
from crosscheck.config import cc_config
import threading
from loguru import logger
import crosscheck.definitions as definitions
import crosscheck.config
from crosscheck.real_time_game import RealTimeGame
import confuse
import pathlib
from typing import Optional
from crosscheck.player.rendering import SimpleImageViewer
from crosscheck.practice.menu import MenuHandler


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
        'save-state': [str, None],
    }),
}


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 practice scenarios')
    parser.add_argument('config', help='App config.')

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
        viewer = SimpleImageViewer(initial_scale=3.5)
        menu = MenuHandler()

        mode = 'cycle'
        if mode == 'full_game':
            minigame = RealTimeGame(button_state, scenarios[0], viewer, menu)
            minigame.play()
        elif mode == 'cycle':
            while True:
                for scenario in scenarios:
                    minigame = RealTimeGame(button_state, scenario, viewer, menu)
                    minigame.play()
        else:
            from crosscheck.practice.minigame import Minigame
            from crosscheck.scorekeeper import string_to_class
            menu = MenuHandler()
            minigame = Minigame('tst', scenarios[0], 10, string_to_class['score-only'],
                                15, button_state, viewer, menu)
            minigame.play()
    finally:
        button_state.running = False



def load_save_state(name: str) -> Optional[pathlib.Path]:
    """
    Load the scenario, and verify the file exists
    :param name: The name of a scenario
    :return: The fully qualified path to the scenario
    """
    if not name:
        return None
    filename = definitions.SAVE_STATE_FOLDER / name
    if not filename.is_file():
        raise FileNotFoundError("Cannot find scenario {}".format(filename))
    return filename





if __name__ == "__main__":
    main(sys.argv[1:])
