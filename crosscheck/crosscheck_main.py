import confuse
import argparse
import multiprocessing
from typing import List, Dict
import neat
import pickle
from loguru import logger
from .config import config
from . import definitions
from . import scorekeeper
from .neat_ import utils as custom_neat_utils


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
        # The 1+ scenarios that are run in parallel
        'scenarios': confuse.Sequence({
            # The filename of a scenario (save state) from which to start play
            'scenario': str,
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
            config.set_file(args.config)
        config.set_args(args, dots=True)

        logger.debug('configuration directory is {}', config.config_dir())

        if config['nproc'].get() is None:
            config['nproc'] = multiprocessing.cpu_count()

        valid_config = config.get(template)
    except confuse.ConfigError as ex:
        logger.critical("Problem parsing config: {}", ex)
        return

    if valid_config['mode'] == 'train':
        train()
    else:
        raise NotImplemented


def train():
    scenarios = load_scenarios()

    # Setup Neat
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                              trainer_class.get_config_filename())

    population = neat.Population(neat_config)

    population.add_reporter(custom_neat_utils.GenerationReporter(True, self.stream))
    population.add_reporter(neat.StatisticsReporter())
    if progress_bar is not None:
        population.add_reporter(custom_neat_utils.TqdmReporter(progress_bar))
    population.add_reporter(custom_neat_utils.Checkpointer(10, stream=self.stream,
                                              filename_prefix=os.path.join(log_folder, "neat-checkpoint-")))
    population.add_reporter(custom_neat_utils.SaveBestOfGeneration(os.path.join(log_folder, "generation-")))

    # Train
    fittest = population.run(_eval_genomes)

    # Dump the result
    with open(model_filename, 'wb') as f:
        pickle.dump(fittest, f, 1)


def _eval_genomes(genomes: List[neat.DefaultGenome], config: neat.Config):
    """
    Evaluate many genomes serially in a for-loop
    """

    for genome_id, genome in genomes:
        stats = _eval_genome(genome, config)
        logger.debug("{gid:5} {score:+5} Stats:{stats}",
                     gid=genome_id, score=genome.fitness, stats=stats)


def _eval_genome(genome: neat.DefaultGenome, config: neat.Config):
    """
    Evaluate a single genome
    :return: The trainer's stats, as a dictionary
    """

    if Runner.genv is None:
        logger.debug("Creating Env")
        self.create_env()
        Runner.genv = self.env
    self.env = Runner.genv

    _ = self.env.reset()

    trainer = self._trainer_class(genome, config, short_circuit=self.short_circuit)

    while not trainer.done:

        self._render()
        next_action = trainer.next_action

        # If there is no action, then no buttons are being pressed
        if next_action is None:
            next_action = [0] * config.genome_config.num_outputs
        step = self.env.step(next_action)

        genome.fitness = trainer.tick(*step, env=self.env)

        for listener in self._listeners:
            listener(*step, {'stats': trainer.stats, 'score_vector': trainer.score_vector})

    self._render()

    return trainer.stats


def load_scenarios() -> List[Dict[str, str]]:
    scenarios = []
    specs = config['input']['scenarios']

    for spec in specs:
        scenario = load_scenario(spec['scenario'].get())

        scorekeeper = load_scorekeeper(spec['scorekeeper'].get())

        scenarios.append({"scenario": scenario, "scorekeeper": scorekeeper})

    return scenarios


def load_scenario(name: str):
    """
    Load the scenario, and verify the file exists
    :param name: The name of a scenario
    :return: The fully qualified path to the scenario
    """
    filename = definitions.SAVE_STATE_FOLDER / name
    if not filename.is_file():
        raise FileNotFoundError("Cannot find scenario {}".format(filename))
    return filename


def load_scorekeeper(name: str) -> scorekeeper.Scorekeeper:
    """
    Load the scorekeeper, and verify that the scorekeeper exists
    :param name: The name of the scorekeeper
    :return: An instance of the scorekeeper
    """
    if name not in scorekeeper.string_to_class:
        raise CrossCheckError(f"Scorekeeper not found: {name} ")
    return scorekeeper.string_to_class[name]


