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

    def __init__(self, trainer_class, render=False, progress_bar=None, stream=print, short_circuit=False):
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

        self.trainer = trainer_class(short_circuit=short_circuit)

        self.render = render
        self.rate = None
        self.env = None

    @property
    def short_circuit(self):
        return self.trainer.short_circuit

    @short_circuit.setter
    def short_circuit(self, value):
        self.trainer.short_circuit = value

    def run(self, genome):
        _ = self.trainer.eval_genome(genome, self.config)

        logger.debug("S:{score:+5} Stats:{stats}",
                    score=genome.fitness, stats=self.trainer.stats)

    def train(self, nproc=1):
        if nproc <= 1:
            self.fittest = self.population.run(self._eval_genomes)
        else:
            parallelizer = neat.ParallelEvaluator(nproc, self._eval_genome)
            self.fittest = self.population.run(parallelizer.evaluate)

    def create_env(self):
        self.env = retro.make('Nhl94-Genesis', 'ChiAtBuf-Faceoff',
                              # obs_type=retro.Observations.RAM,
                              inttype=retro.data.Integrations.ALL)

        self.env = self.trainer.discretizer(self.env)

    def _eval_genome(self, genome, config):

        if FaceoffTrainer.genv is None:
            logger.warning("Creating Env")
            self.create_env()
            Runner.genv = self.env
        self.env = Runner.genv

        _ = self.env.reset()

        self.trainer.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        while not self.trainer.done:

            if self.render:
                _ = self.env.render()
                if self.rate:
                    time.sleep(0.01)

            step = self.env.step(self.trainer.next_action)

            genome.fitness = self.trainer.tick(*step)

    def _eval_genomes(self, genomes, config):

        for genome_id, genome in genomes:
            _ = self.trainer.eval_genome(genome, config)
            logger.debug("{gid:5} {score:+5} Stats:{stats}",
                    gid=genome_id, score=genome.fitness, stats=self.trainer.stats)


def main(argv, trainer_class):
    parser = argparse.ArgumentParser(description="Train or test a model to win a faceoff")
    parser.add_argument('--render', action='store_true', help="Watch the game while training")
    parser.add_argument('--replay', action='store_true', help="Replay a trained network")
    parser.add_argument('--model-file', type=str, nargs=1, help="model file for input (replay) or output (train)",
                        default=None)
    parser.add_argument('--nproc', type=int, help="The number of processes to run", default=multiprocessing.cpu_count())
    args = parser.parse_args(argv)
    model_filename = None

    module_name = os.path.splitext(os.path.basename(__file__))[0]
    log_folder_root = os.path.abspath(os.path.join("logs", module_name))

    if not args.replay:
        # Setup target og folder
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
            trainer.short_circuit = False

            # Replay
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
            trainer.run(model)
