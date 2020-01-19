import confuse
import argparse
import multiprocessing
import pathlib
from typing import List, Callable
from loguru import logger
from .config import cc_config
from . import definitions
from . import scorekeeper
from . import metascorekeeper
from .info_utils.feature_vector import string_to_class as feature_vector_string_to_class
from .neat_.trainer import Trainer
from .scenario import Scenario


class CrossCheckError(Exception):
    pass

template = {
    # The mode in which to run
    #  'train': Create a new model
    #  'replay': Run an existing model
    #  'compete': Run two models against each other
    #  'play': Run a model against a human player (controller input)
    #  'play-2p': Play traditional human vs human
    'mode': confuse.OneOf(['train', 'replay', 'compete', 'play', 'play-2p']),
    'input': {
        'feature-vector': str,
        # The 1+ scenarios that are run in parallel
        'scenarios': confuse.Sequence({
            # The name of a scenario
            'name': str,
            # The filename of a scenario (save state) from which to start play
            'save-state': str,
            # How play in this scenario will be judged
            'scorekeeper': str,
        }),
        # The cost function to use when combining scenarios
        'metascorekeeper': str,
    },
    # Information about the movie to record
    'movie': {
        # When to create a movie
        # 'generations': The best of each generation (unless duplicated)
        # 'latest': Only the best of the latest generation
        # 'off': No movie
        'enabled': confuse.OneOf(['generations', 'latest', 'off']),
        # How long to keep playing the movie after the scenario says it is done
        'stoppage-time-s': float,
    },
    # True to show the movie in a window during execution
    'render-live': bool,
    # The number of processes to run
    'nproc': int,
}


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 reinforcement learning')
    parser.add_argument('--config', '-c', help='App config.')
    parser.add_argument('--nproc', type=int,
                        dest="nproc",
                        help="The number of processes to run")
    parser.add_argument('--latest-only',
                        action='store_const',
                        const="latest",
                        dest="movie.enabled",
                        help="Only replay the latest training model")

    args = parser.parse_args(argv)

    # Parse the config using confuse, bailing on failure
    try:

        if args.config:
            cc_config.set_file(args.config)
        cc_config.set_args(args, dots=True)

        logger.debug('configuration directory is {}', cc_config.config_dir())

        if cc_config['nproc'].get() is None:
            cc_config['nproc'] = multiprocessing.cpu_count()

        valid_config = cc_config.get(template)
    except confuse.ConfigError as ex:
        logger.critical("Problem parsing config: {}", ex)
        return

    if valid_config['mode'] == 'train':
        train()
    else:
        raise NotImplemented


def train():
    feature_vector = load_feature_vector(cc_config['input']['feature-vector'].get())
    scenarios = load_scenarios(cc_config['input']['scenarios'])
    combiner = load_metascorekeeper(cc_config['input']['metascorekeeper'].get())
    trainer = Trainer(scenarios, combiner, feature_vector)
    trainer.train()


def load_scenarios(specs: dict) -> List[Scenario]:
    """
    Convert config to a list of Scenario object
    :param specs: The config for a scenario
    :return: List of scenario objects
    """
    scenarios = [Scenario(
        name=spec['name'].get(),
        save_state=load_save_state(spec['save-state'].get()),
        scorekeeper=load_scorekeeper(spec['scorekeeper'].get()))
                 for spec in specs]

    return scenarios


def load_feature_vector(name: str) -> Callable[[dict], List[float]]:
    """
    Load the feature vector, and verify that the it exists
    :param name: The name of the feature vector
    :return: An instance of the feature vector
    """
    if name not in feature_vector_string_to_class:
        raise CrossCheckError(f"Feature vector not found: {name} ")
    return feature_vector_string_to_class[name]


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


def load_scorekeeper(name: str) -> Callable[[],scorekeeper.Scorekeeper]:
    """
    Load the scorekeeper, and verify that the scorekeeper exists
    :param name: The name of the scorekeeper
    :return: An instance of the scorekeeper
    """
    if name not in scorekeeper.string_to_class:
        raise CrossCheckError(f"Scorekeeper not found: {name} ")
    return scorekeeper.string_to_class[name]


def load_metascorekeeper(name: str) -> Callable[[], metascorekeeper.Metascorekeeper]:
    """
    Load the metascorekeeper, and verify that the metascorekeeper exists
    :param name: The name of the metascorekeeper
    :return: An instance of the metascorekeeper
    """
    if name not in metascorekeeper.string_to_class:
        raise CrossCheckError(f"Metascorekeeper not found: {name} ")
    return metascorekeeper.string_to_class[name]