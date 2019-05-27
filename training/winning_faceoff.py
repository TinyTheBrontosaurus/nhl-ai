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


class Discretizer(gym.ActionWrapper):
    """
    Simplify inputs to win a faceoff
    """
    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        # Values are:
        #  ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        buttons = env.buttons

        # Create all combinations on the D pad for pressing B or not (2*3*3==18)
        actions = []
        for button in ['B', None]:
            for lr in ['LEFT', 'RIGHT', None]:
                for ud in ['UP', 'DOWN', None]:
                    action = []
                    if button:
                        action.append(button)
                    if lr:
                        action.append(lr)
                    if ud:
                        action.append(ud)
                    actions.append(action)

        # Convert easy-to-debug strings to boolean array
        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        index = math.floor(a[0] * len(self._actions))
        if index >= len(self._actions):
            index = len(self._actions) - 1
        return self._actions[index].copy()



class FaceoffTrainerRunner:

    def __init__(self, render=False, progress_bar=None, stream=print, short_circuit=False):
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

        self.trainer = FaceoffTrainer(render, short_circuit=short_circuit)


    @property
    def rate(self):
        return self.trainer.rate

    @rate.setter
    def rate(self, value):
        self.trainer.rate = value

    @property
    def render(self):
        return self.trainer.render

    @render.setter
    def render(self, value):
        self.trainer.render = value

    @property
    def short_circuit(self):
        return self.trainer.short_circuit

    @short_circuit.setter
    def short_circuit(self, value):
        self.trainer.short_circuit = value

    def run(self, genome):
        _ = self.trainer.eval_genome(genome, self.config)
        results = self.trainer.results

        logger.debug("S:{score:+5} T:{counter} Y:{puck_y:+4}, #:{player_w_puck}",
                    score=genome.fitness, counter=results["frame"], puck_y=results["puck_y"],
                    player_w_puck=results["player_w_puck"])

    def train(self, nproc=1):
        if nproc <= 1:
            self.fittest = self.population.run(self.eval_genomes)
        else:
            parallelizer = neat.ParallelEvaluator(nproc, self.trainer.eval_genome)
            self.fittest = self.population.run(parallelizer.evaluate)

    def eval_genomes(self, genomes, config):

        for genome_id, genome in genomes:

            _ = self.trainer.eval_genome(genome, config)
            results = self.trainer.results
            logger.debug("{gid:5} {score:+5} T:{counter} Y:{puck_y:+4}, #:{player_w_puck}",
                        gid=genome_id, score=genome.fitness, counter=results["frame"], puck_y=results["puck_y"],
                        player_w_puck=results["player_w_puck"])


class FaceoffTrainer:
    genv = None
    def __init__(self, render=False, short_circuit=False):
        self.env = None
        self.render = render
        self.rate = None
        self.results = None
        self.short_circuit = short_circuit

    def create_env(self):
        self.env = retro.make('Nhl94-Genesis', 'ChiAtBuf-Faceoff',
                              obs_type=retro.Observations.RAM,
                              inttype=retro.data.Integrations.ALL)

        self.env = Discretizer(self.env)

    def eval_genome(self, genome, config):

            if FaceoffTrainer.genv is None:
                logger.warning("Creating Env")
                self.create_env()
                FaceoffTrainer.genv = self.env
            self.env = FaceoffTrainer.genv

            _ = self.env.reset()

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            frame = 0
            score = 0

            done = False

            min_puck_y = 500
            puck_y = 0
            player_w_puck = None

            while not done:

                frame += 1

                features = [frame]
                actions = net.activate(features)

                if self.render:
                    _ = self.env.render()
                    if self.rate:
                        time.sleep(0.01)

                ob, rew, done, info = self.env.step(actions)

                faceoffs_won = info['home-faceoff']
                faceoffs_lost = info['away-faceoff']
                puck_y = info['puck-ice-y']
                total_faceoffs = faceoffs_won + faceoffs_lost
                min_puck_y = min(puck_y, min_puck_y)

                if info['player-w-puck-ice-x'] == info['player-home-7-x'] and \
                   info['player-w-puck-ice-y'] == info['player-home-7-y']:
                    player_w_puck = 7
                    puck_bonus = 1000
                elif info['player-w-puck-ice-x'] == info['player-home-10-x'] and \
                     info['player-w-puck-ice-y'] == info['player-home-10-y']:
                    player_w_puck = 10
                    puck_bonus = 200
                elif info['player-w-puck-ice-x'] == info['player-home-16-x'] and \
                     info['player-w-puck-ice-y'] == info['player-home-16-y']:
                    player_w_puck = 16
                    puck_bonus = 300
                elif info['player-w-puck-ice-x'] == info['player-home-89-x'] and \
                     info['player-w-puck-ice-y'] == info['player-home-89-y']:
                    player_w_puck = 89
                    puck_bonus = 500
                elif info['player-w-puck-ice-x'] == info['player-home-8-x'] and \
                     info['player-w-puck-ice-y'] == info['player-home-8-y']:
                    player_w_puck = 8
                    puck_bonus = 100
                elif info['player-w-puck-ice-x'] == info['player-home-goalie-ice-x'] and \
                     info['player-w-puck-ice-y'] == info['player-home-goalie-ice-y']:
                    player_w_puck = 31
                    puck_bonus = 20
                else:
                    player_w_puck = None
                    puck_bonus = 0

                if faceoffs_won <= 0:
                    puck_bonus = 0

                if frame > 300:
                    done = True

                if self.short_circuit and puck_bonus > 0 and faceoffs_won > 0:
                    done = True

                score = (faceoffs_won - faceoffs_lost) * 100 + -min_puck_y + puck_bonus

                genome.fitness = score

            self.results = {"score": score, "frame": frame, "puck_y": puck_y, "player_w_puck": player_w_puck}
            return score


def main():
    parser = argparse.ArgumentParser(description="Train or test a model to win a faceoff")
    parser.add_argument('--render', action='store_true', help="Watch the game while training")
    parser.add_argument('--replay', action='store_true', help="Replay a trained network")
    parser.add_argument('--model-file', type=str, nargs=1, help="model file for input (replay) or output (train)",
                        default=None)
    parser.add_argument('--nproc', type=int, help="The number of processes to run", default=multiprocessing.cpu_count())
    args = parser.parse_args()
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
            for folder in reversed(sorted(potential_folders)):
                if os.path.isdir(folder):
                    model_filename = os.path.join(folder, "fittest.pkl")
                    if os.path.exists(model_filename):
                        logger.info("Replaying {}".format(folder))
                        break
            if model_filename is None:
                raise FileNotFoundError("Could not find fittest.pkl")

    # Run tqdm and do training vs replay
    with tqdm.tqdm(smoothing=0, unit='generation') as progress_bar:
        trainer = FaceoffTrainerRunner(render=args.render, progress_bar=progress_bar, stream=logger.info)

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



if __name__ == "__main__":
    main()
