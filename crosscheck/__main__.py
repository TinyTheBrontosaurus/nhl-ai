import confuse
import argparse
import multiprocessing
import sys
from loguru import logger


template = {
    'mode': confuse.OneOf(['train', 'replay', 'compete']),
    'input': {
        'scenarios': confuse.Sequence({
            'scenario': str,
            'scorekeeper': str,
        }),
        'metascorekeeper': str,
    },
    'movie': {
        'enabled': confuse.OneOf(['all', 'generations', 'latest', 'off']),
        'stoppage-time-s': float,
    },
    'render-live': bool,
    'nproc': int,
}

config = confuse.LazyConfig('Cross-check: NHL \'94 reinforcement learning', __name__)


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check')
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

    if args.config:
        config.set_file(args.config)
    config.set_args(args, dots=True)

    logger.debug('configuration directory is', config.config_dir())

    if config['nproc'].get() is None:
        config['nproc'] = multiprocessing.cpu_count()

    valid = config.get(template)


if __name__ == "__main__":
    main(sys.argv[1:])
