import neat
import retro
import pickle
import gym
import argparse
from loguru import logger
import numpy as np
import math
import time
import tqdm
import multiprocessing
import subprocess
import os
import datetime
from training.custom_neat_utils import TqdmReporter, GenerationReporter, Checkpointer


class Runner:

    genv = None

    def __init__(self, trainer_class, render=False, progress_bar=None, stream=print, short_circuit=False):
        """
        Provides a progress bar, logging to files assorted by type and timestamp.
        :param trainer_class: The class to be used for training
        :param render: True to render the emulated screen, which will decrease speed
        :param progress_bar: The progress bar to use
        :param stream: The stream to which to print logs
        :param short_circuit: True to indicate to the trainer to stop as soon as the goal is achieved. Typically
        don't want to short-circuit if viewing a replay
        """
        self.stream = stream
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  'config-feedforward')

        self.population = neat.Population(self.config)

        self.population.add_reporter(GenerationReporter(True, self.stream))
        self.population.add_reporter(neat.StatisticsReporter())
        if progress_bar is not None:
            self.population.add_reporter(TqdmReporter(progress_bar, stream=logger.info))
        self.population.add_reporter(Checkpointer(10, stream=self.stream))

        self.fittest = None

        self._trainer_class = trainer_class

        self.render = render
        self.rate = None
        self.env = None
        self._short_circuit = short_circuit

    def run(self, genome):
        """
        Replay against a specific genome
        """
        stats = self._eval_genome(genome, self.config)

        logger.debug("S:{score:+5} Stats:{stats}",
                    score=genome.fitness, stats=stats)

    def train(self, nproc=1):
        """
        Train with a specified number of processes
        :param nproc: The number of processes
        """
        if nproc <= 1:
            # Single-threaded. Print more stats
            self.fittest = self.population.run(self._eval_genomes)
        else:
            # Multi-threaded. Use the parallelizer and print fewer stats
            parallelizer = neat.ParallelEvaluator(nproc, self._eval_genome)
            self.fittest = self.population.run(parallelizer.evaluate)

    def create_env(self):
        """
        Create the environment. This is created once per process to save CPU
        """
        self.env = retro.make('Nhl94-Genesis', 'ChiAtBuf-Faceoff',
                              obs_type=retro.Observations.RAM,
                              inttype=retro.data.Integrations.ALL)

        self.env = self._trainer_class.discretizer_class()(self.env)

    def _eval_genomes(self, genomes, config):
        """
        Evaluate many genomes serially in a for-loop
        """

        for genome_id, genome in genomes:
            stats = self._eval_genome(genome, config)
            logger.debug("{gid:5} {score:+5} Stats:{stats}",
                    gid=genome_id, score=genome.fitness, stats=stats)

    def _eval_genome(self, genome, config):
        """
        Evaluate a single genome
        :return: The trainer's stats, as a dictionary
        """

        if Runner.genv is None:
            logger.warning("Creating Env")
            self.create_env()
            Runner.genv = self.env
        self.env = Runner.genv

        _ = self.env.reset()

        trainer = self._trainer_class(genome, config, short_circuit=self._short_circuit)

        while not trainer.done:

            self._render()
            step = self.env.step(trainer.next_action)

            genome.fitness = trainer.tick(*step)

        self._render()

        return trainer.stats

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
    parser.add_argument('--render', action='store_true', help="Watch the game while running")
    parser.add_argument('--replay', action='store_true', help="Replay a trained network")
    parser.add_argument('--model-file', type=str, nargs=1, help="model file for input (replay) or output (train)",
                        default=None)
    parser.add_argument('--nproc', type=int, help="The number of processes to run", default=multiprocessing.cpu_count())
    args = parser.parse_args(argv)
    model_filename = None

    module_name = trainer_class.__name__
    log_folder_root = os.path.abspath(os.path.join("logs", module_name))

    if not args.replay:
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
    else:
        # If replaying, then find the most recent logged folder with a fittest.pkl
        if args.model_file is None:
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

    # Run tqdm and do training vs replay
    with tqdm.tqdm(smoothing=0, unit='generation') as progress_bar:
        trainer = Runner(trainer_class=trainer_class,
                         render=args.render,
                         progress_bar=progress_bar,
                         stream=logger.info)

        if not args.replay:
            # Train
            trainer.train(nproc=args.nproc)

            with open(model_filename, 'wb') as f:
                pickle.dump(trainer.fittest, f, 1)
        else:
            # Make it easy to view when replaying
            trainer.rate = 1
            trainer.render = True
            trainer._short_circuit = False

            # Replay
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
            trainer.run(model)
