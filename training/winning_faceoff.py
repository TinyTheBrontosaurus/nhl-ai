import neat
import retro
import pickle
import gym
from loguru import logger
import numpy as np
import math



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

class FaceoffTrainer:

    def __init__(self):
        self.env = retro.make('Nhl94-Genesis', 'ChiAtBuf-Faceoff',
                              obs_type=retro.Observations.RAM,
                              inttype=retro.data.Integrations.ALL)

        self.env = Discretizer(self.env)
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  'config-feedforward')

        self.population = neat.Population(self.config)

        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())
        self.population.add_reporter(neat.Checkpointer(10))

        self.fittest = None

    def run(self, genome):
        _ = self.env.reset()

        frame = 0
        counter = 0
        score = 0

        done = False

        while not done:

            frame += 1
            counter += 1

            self.env.render()

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, self.config)

            features = [frame]
            actions = net.activate(features)

            ob, rew, done, info = self.env.step(actions)

            faceoffs_won = info['home-faceoff']
            faceoffs_lost = info['away-faceoff']
            total_faceoffs = faceoffs_won + faceoffs_lost

            if counter > 600:
                done = True

            if total_faceoffs >= 1:
                done = True

            score = faceoffs_won - faceoffs_lost

        logger.info("{score:+} {counter}",
                    score=score, counter=counter)

    def train(self):
        self.fittest = self.population.run(self._eval_genomes)

    def _eval_genomes(self, genomes, config):

        for genome_id, genome in genomes:
            _ = self.env.reset()

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            frame = 0
            counter = 0

            done = False

            while not done:

                frame += 1
                counter += 1

                features = [frame]
                actions = net.activate(features)

                ob, rew, done, info = self.env.step(actions)

                faceoffs_won = info['home-faceoff']
                faceoffs_lost = info['away-faceoff']
                total_faceoffs = faceoffs_won + faceoffs_lost

                if counter > 600:
                    done = True

                if total_faceoffs >= 1:
                    done = True

                genome.fitness = faceoffs_won - faceoffs_lost

            logger.info("{gid:3} {score:+} {counter}",
                        gid=genome_id, score=genome.fitness, counter=counter)



def main():
    trainer = FaceoffTrainer()
    with open('fittest.pkl', 'rb') as f:
        model = pickle.load(f)

    trainer.run(model)

    logger.info("Starting Training")
    trainer.train()

    with open('fittest.pkl', 'wb') as output:
        pickle.dump(trainer.fittest, output, 1)


if __name__ == "__main__":
    main()
