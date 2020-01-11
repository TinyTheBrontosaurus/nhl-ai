import confuse
import argparse
import multiprocessing
import sys
from loguru import logger


template = {
    # The mode in which to run
    #  'train': Create a new model
    #  'replay': Run an existing model
    #  'compete': Run two models against each other
    #  'play': Run a model against a human player (controller input)
    #  'play-2p': Play traditional human vs human
    'mode': confuse.OneOf(['train', 'replay', 'compete', 'play', 'play-2p']),
    'input': {
        # The 1+ scenarios that are run in parallel
        'scenarios': confuse.Sequence({
            'scenario': str,
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

config = confuse.LazyConfig('cross-check', __name__)


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
            config.set_file(args.config)
        config.set_args(args, dots=True)

        logger.debug('configuration directory is {}', config.config_dir())

        if config['nproc'].get() is None:
            config['nproc'] = multiprocessing.cpu_count()

        valid_config = config.get(template)
    except confuse.ConfigError as ex:
        logger.critical("Problem parsing config: {}", ex)
        return







if __name__ == "__main__":
    main(sys.argv[1:])
