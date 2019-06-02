import neat
import retro
import pickle
import gym
import argparse
from loguru import logger
import time
import tqdm
import multiprocessing
import subprocess
import os
import datetime
from training.custom_neat_utils import TqdmReporter, GenerationReporter, Checkpointer, SaveBestOfGeneration
import abc
import typing
from natsort import natsorted


def logger_info_workaround(*args, **kwargs):
    """
    Due to a bug in loguru/multiprocessing interaction, it is not possible to send a logger.info
    function pointer into multiprocessing to be picked. A workaround is to wrap in another
    function
    See https://github.com/Delgan/loguru/issues/102 for bug details and resolution
    """
    return logger.info(*args, **kwargs)


class Trainer(abc.ABC):
    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, short_circuit: bool):
        """
        Abstract class for scoring scenarios and running genomes
        :param genome: The genome
        :param config: The config
        :param short_circuit: False to keep running to the end of the scenario. True to stop as soon as it's won/lost
        """

        self.genome = genome
        self.config = config
        self.short_circuit = short_circuit

    @classmethod
    def get_config_filename(cls) -> str:
        """
        Accessor for config filename
        """
        return 'config-feedforward'

    @classmethod
    def get_scenario_string(cls) -> str:
        """
        Accessor for the scenario to use
        """
        return 'ChiAtBuf-Faceoff'

    @abc.abstractclassmethod
    def discretizer_class(cls) -> typing.Callable[[], gym.ActionWrapper]:
        """
        Return a discretizer ctor that will be instantiated for this object
        :return:
        """

    @abc.abstractmethod
    def next_action(self) -> list:
        """
        Property that returns the next action to pass into the env
        """

    @abc.abstractmethod
    def stats(self) -> dict:
        """
        Property that returns the stats of the trainer as a dictionary
        """

    @abc.abstractmethod
    def done(self) -> bool:
        """
        Property that returns true when done running
        """

    @abc.abstractmethod
    def tick(self, ob, rew, done, info) -> float:
        """
        Process a single timestep of the emulator. Expected that given this latest state,
        the actions and score will be updated
        :param ob: The observation (
        :param rew: The reward
        :param done: Whether done
        :param info: Emulator memory parsed into a dictionary
        :return: Score
        """

class Runner:

    genv = None

    def __init__(self, trainer_class: typing.Type[Trainer],
                 render:bool=False,
                 progress_bar:tqdm.tqdm=None,
                 stream=print,
                 short_circuit:bool=False,
                 log_folder:str=""):
        """
        Provides a progress bar, logging to files assorted by type and timestamp.
        :param trainer_class: The class to be used for training
        :param render: True to render the emulated screen, which will decrease speed
        :param progress_bar: The progress bar to use
        :param stream: The stream to which to print logs
        :param short_circuit: True to indicate to the trainer to stop as soon as the goal is achieved. Typically
        don't want to short-circuit if viewing a replay
        :param log_folder: The folder in which to log checkpoints and generational results
        """
        self.stream = stream
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  trainer_class.get_config_filename())

        self.population = neat.Population(self.config)

        self.population.add_reporter(GenerationReporter(True, self.stream))
        self.population.add_reporter(neat.StatisticsReporter())
        if progress_bar is not None:
            self.population.add_reporter(TqdmReporter(progress_bar, stream=logger_info_workaround))
        self.population.add_reporter(Checkpointer(10, stream=self.stream,
                                                  filename_prefix=os.path.join(log_folder, "neat-checkpoint-")))
        self.population.add_reporter(SaveBestOfGeneration(os.path.join(log_folder, "generation-")))

        self.fittest = None

        self._trainer_class = trainer_class

        self.render = render
        self.rate = None
        self.env = None
        self._short_circuit = short_circuit
        self._listeners: typing.Callable[[typing.Any] * 4 + [dict], typing.Any] = []

    def replay(self, genome: neat.DefaultGenome):
        """
        Replay against a specific genome
        """
        stats = self._eval_genome(genome, self.config)

        logger.debug("S:{score:+5} Stats:{stats}",
                    score=genome.fitness, stats=stats)

    def train(self, nproc:int=1):
        """
        Train with a specified number of processes
        :param nproc: The number of processes
        """
        # Run single-threaded is as follows. Kept in for easier debugging
        if nproc <= 1:
            self.fittest = self.population.run(self._eval_genomes)
        else:
            # Multi-threaded execution
            parallelizer = neat.ParallelEvaluator(nproc, self._eval_genome_score)
            self.fittest = self.population.run(parallelizer.evaluate)

    def create_env(self):
        """
        Create the environment. This is created once per process to save CPU
        """
        self.env = retro.make('Nhl94-Genesis', self._trainer_class.get_scenario_string(),
                              inttype=retro.data.Integrations.ALL)

        # Wrap the env
        if self._trainer_class.discretizer_class is not None:
            self.env = self._trainer_class.discretizer_class()(self.env)

    def add_listener(self, listener: typing.Callable[[typing.Any] * 4 + [dict], typing.Any]):
        """
        Add a listener callback that is called with a frame and all the stats/reward/image associated
        that that frame
        """
        self._listeners.append(listener)

    def _eval_genomes(self, genomes: typing.List[neat.DefaultGenome], config: neat.Config):
        """
        Evaluate many genomes serially in a for-loop
        """

        for genome_id, genome in genomes:
            stats = self._eval_genome(genome, config)
            logger.debug("{gid:5} {score:+5} Stats:{stats}",
                    gid=genome_id, score=genome.fitness, stats=stats)

    def _eval_genome(self, genome: neat.DefaultGenome, config: neat.Config):
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

        trainer = self._trainer_class(genome, config, short_circuit=self._short_circuit)

        while not trainer.done:

            self._render()
            next_action = trainer.next_action

            # If there is no action, then no buttons are being pressed
            if next_action is None:
                next_action = [0] * config.genome_config.num_outputs
            step = self.env.step(next_action)

            genome.fitness = trainer.tick(*step)

            for listener in self._listeners:
                listener(*step, trainer.stats)

        self._render()

        return trainer.stats

    def _eval_genome_score(self, genome: neat.DefaultGenome, config: neat.Config):
        _ = self._eval_genome(genome, config)
        return genome.fitness

    def _render(self):
        """
        Render and sleep based upon the settings
        """
        if self.render:
            _ = self.env.render()
            if self.rate:
                time.sleep(0.01)


def main(argv, trainer_class):
    parser = argparse.ArgumentParser(description="Train or test a model to win a faceoff")
    replay_parser = parser.add_mutually_exclusive_group()
    replay_parser.add_argument('--render', action='store_true', help="Watch the game while training")
    replay_parser.add_argument('--replay', action='store_true',
                               help="Replay a trained network")
    replay_parser.add_argument('--replay-training', action='store_true',
                               help="Replay all major generations of training")
    parser.add_argument('--model-file', type=str, nargs=1, help="model file for input (replay) or output (train)",
                        default=None)
    parser.add_argument('--nproc', type=int, help="The number of processes to run", default=multiprocessing.cpu_count())
    args = parser.parse_args(argv)
    args.model_filename = args.model_file[0] if args.model_file is not None else None

    module_name = trainer_class.__name__
    args.log_folder_root = os.path.abspath(os.path.join("logs", module_name))
    args.trainer_class = trainer_class

    do_replay = args.replay or args.replay_training

    if not do_replay:
        model_filename = args.model_filename
        log_folder_root = args.log_folder_root

        # Setup target log folder
        module_name = os.path.splitext(os.path.basename(__file__))[0]
        friendly_time = str(datetime.datetime.now()).replace(':', "-").replace(" ", "_")
        log_folder = os.path.join(log_folder_root, friendly_time)

        # Replace the standard logger with logging to a file
        logger.remove(0)
        logger.add(os.path.join(log_folder, "event.log"))

        if args.model_file is None:
            model_filename = os.path.join(log_folder, "fittest.pkl")
        # Log the beginning of the file
        version = subprocess.check_output(["git", "describe", "--dirty", "--always"]).strip()

        logger.info("Running program: {}", module_name)
        logger.info("Version: {}", version)
        logger.info("Full path: {}", __file__)

        # Run tqdm and do training
        with tqdm.tqdm(smoothing=0, unit='generation') as progress_bar:
            runner = Runner(trainer_class=trainer_class,
                            render=args.render,
                            progress_bar=progress_bar,
                            stream=logger_info_workaround,
                            log_folder=log_folder)

            # Train
            runner.train(nproc=args.nproc)

        # Dump the result
        with open(model_filename, 'wb') as f:
            pickle.dump(runner.fittest, f, 1)

    elif args.replay:
        replay(args)
    elif args.replay_training:
        replay_training(args)

def replay(args):
    """
    Search for the latest folder with a "fittest" genome in it, and replay it
    If a genome file is provided, uses that instead of searching
    """
    model_filename = args.model_filename
    log_folder_root = args.log_folder_root

    # If replaying, then find the most recent logged folder with a fittest.pkl
    if model_filename is None:
        potential_folders = os.listdir(log_folder_root)
        for subfolder in reversed(sorted(potential_folders)):
            folder = os.path.join(log_folder_root, subfolder)
            if os.path.isdir(folder):
                potential_filename = os.path.join(folder, "fittest.pkl")
                if os.path.exists(potential_filename):
                    model_filename = potential_filename
                    logger.info("Replaying {}".format(folder))
                    break
        if model_filename is None:
            raise FileNotFoundError("Could not find fittest.pkl")

    # Replay
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)

    runner = Runner(trainer_class=trainer_class,
                    render=True,
                    stream=logger_info_workaround,
                    short_circuit=False)

    # Make it easy to view when replaying
    runner.rate = 1

    runner.replay(model)


def replay_training(args):
    """
    Search for the latest folder with genome logs in it, and replay those genomes
    """
    log_folder_root = args.log_folder_root

    model_filenames = None

    # If replaying, then find the most recent logged folder with any generation-{}.pkl
    potential_folders = os.listdir(log_folder_root)
    for subfolder in reversed(sorted(potential_folders)):
        folder = os.path.join(log_folder_root, subfolder)
        if os.path.isdir(folder):
            potential_filename = os.path.join(folder, "generation-0.pkl")
            if os.path.exists(potential_filename):
                model_filenames = [os.path.join(folder, x) for x in os.listdir(folder)
                                   if x.startswith('generation-') and x.endswith('.pkl')]

                logger.info("Replaying {} unique generations from training in folder {}".format(
                    folder, len(model_filenames)))
                break
    if model_filenames is None:
        raise FileNotFoundError("Could not find fittest.pkl")

    # Replay
    models = []
    for model_filename in natsorted(model_filenames):
        with open(model_filename, 'rb') as f:
            models.append( pickle.load(f))

    runner = Runner(trainer_class=args.trainer_class,
                    render=True,
                    stream=logger_info_workaround,
                    short_circuit=False)

    # Make it easy to view when replaying
    runner.rate = 1
    runner.add_listener(movie_maker)

    for model in models:
        runner.replay(model)

def movie_maker(ob, rew, done, info, stats):
    bar = 1