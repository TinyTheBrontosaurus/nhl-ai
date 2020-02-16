import confuse
import argparse
import multiprocessing
import pathlib
import shutil
from typing import List, Callable, Type, Optional
from loguru import logger
from .config import cc_config
import crosscheck.config
from . import definitions
from . import scorekeeper
from . import metascorekeeper
from . import discretizers
from .info_utils.feature_vector import string_to_class as feature_vector_string_to_class
from .neat_.trainer import Trainer
from .scenario import Scenario
from .log_folder import LogFolder
from .version import __version__
import natsort


class CrossCheckError(Exception):
    pass

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
    'input': {
        'feature-vector': str,
        'controller-discretizer': str,
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
        # Custom configs that are applied directly to the neat.ini files
        'neat-config': dict,
        # The checkpoint file to load (relative to yaml file)
        'load-checkpoint': [{
            'latest': bool,
            'specific-filename': [str, None]
        }, None],
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

        crosscheck.config.filename = args.config

    except confuse.ConfigError as ex:
        logger.critical("Problem parsing config: {}", ex)
        return

    # Cache old log path
    crosscheck.config.log_name = cc_config['name'].get()
    try:
        LogFolder.get_latest_log_folder(definitions.LOG_ROOT / crosscheck.config.log_name)
    except FileNotFoundError:
        pass

    # Create log path
    LogFolder.set_path(definitions.LOG_ROOT, crosscheck.config.log_name)

    # Initialize logger
    logger.add(LogFolder.folder / "event.log", level="INFO")
    logger.info("Running program: {}", cc_config['name'].get())
    logger.info("Version: {}", __version__)
    logger.info("Log folder: {}", LogFolder.folder)
    checkpoint_filename = load_checkpoint_filename(cc_config['input']['load-checkpoint'].get())
    logger.info(f"Checkpoint file: {str(checkpoint_filename)}")

    # Don't lost to stderr anymore
    logger.remove(0)

    # Copy configs
    # Straight copy
    orig_config = pathlib.Path(args.config)
    shutil.copyfile(str(orig_config), str(LogFolder.folder / orig_config.name))
    logger.info("CLI args: {}", argv)

    # Copy resulting config
    with open(LogFolder.folder / "config.yml", 'w') as f:
        f.write(cc_config.dump())

    # Start program
    if valid_config['mode'] == 'train':
        train()
    else:
        raise NotImplemented


def train():
    discretizer = load_discretizer(cc_config['input']['controller-discretizer'].get())
    feature_vector = load_feature_vector(cc_config['input']['feature-vector'].get())
    scenarios = load_scenarios(cc_config['input']['scenarios'])
    combiner = load_metascorekeeper(cc_config['input']['metascorekeeper'].get())
    checkpoint_filename = load_checkpoint_filename(cc_config['input']['load-checkpoint'].get())
    trainer = Trainer(scenarios, combiner, feature_vector, cc_config['input']['neat-config'],
                      discretizer, nproc=cc_config['nproc'].get(), checkpoint_filename=checkpoint_filename)
    trainer.train()


def load_checkpoint_filename(specs: dict) -> Optional[pathlib.Path]:
    # Setup the checkpoint filename so that it is an absolute path
    # If it comes in as a relative path, resolve it as being originally relative to the config file path
    if specs is None:
        return None

    if not specs.get('latest'):
        specific_filename = specs.get('specific-file')
        if not specific_filename:
            return None

        rel_checkpoint_filename = pathlib.Path(specific_filename)
        if not rel_checkpoint_filename.is_absolute():
            return (pathlib.Path(crosscheck.config.filename).parent / rel_checkpoint_filename).resolve()
    else:
        try:
            checkpoints_folder = LogFolder.get_latest_log_folder(definitions.LOG_ROOT / crosscheck.config.log_name) / "checkpoints"
            checkpoints = natsort.natsorted(list(checkpoints_folder.iterdir()))
            return checkpoints_folder / checkpoints[-1]
        except FileNotFoundError:
            return None


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


def load_discretizer(name: str) -> Type[discretizers.Independent]:
    if name not in discretizers.string_to_class:
        raise CrossCheckError(f"Discretizer not found: {name} ")
    return discretizers.string_to_class[name]


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


def load_scorekeeper(name: str) -> Type[scorekeeper.Scorekeeper]:
    """
    Load the scorekeeper, and verify that the scorekeeper exists
    :param name: The name of the scorekeeper
    :return: An instance of the scorekeeper
    """
    if name not in scorekeeper.string_to_class:
        raise CrossCheckError(f"Scorekeeper not found: {name} ")
    return scorekeeper.string_to_class[name]


def load_metascorekeeper(name: str) -> Type[metascorekeeper.Metascorekeeper]:
    """
    Load the metascorekeeper, and verify that the metascorekeeper exists
    :param name: The name of the metascorekeeper
    :return: An instance of the metascorekeeper
    """
    if name not in metascorekeeper.string_to_class:
        raise CrossCheckError(f"Metascorekeeper not found: {name} ")
    return metascorekeeper.string_to_class[name]