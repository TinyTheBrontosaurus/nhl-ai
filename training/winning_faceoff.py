import neat
import retro
import pickle
from loguru import logger


class FaceoffTrainer:

    def __init__(self):
        self.env = retro.make('Nhl94-Genesis', 'ChiAtBuf-Faceoff',
                              obs_type=retro.Observations.RAM,
                              inttype=retro.data.Integrations.ALL)

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  'config-feedforward')

        self.population = neat.Population(self.config)

        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())
        self.population.add_reporter(neat.Checkpointer(10))

        self.fittest = None

    def train(self):
        self.fittest = self.population.run(self._eval_genomes)

    def _eval_genomes(self, genomes, config):

        for genome_id, genome in genomes:
            ob = self.env.reset()
            actions = self.env.action_space.sample()

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            current_max_fitness = 0
            fitness_current = 0
            frame = 0
            counter = 0

            done = False

            while not done:

                frame += 1

                features = range(15)
                actions = net.activate(features)

                ob, rew, done, info = self.env.step(actions)

                faceoffs_won = info['home-faceoff']
                faceoffs_lost = info['away-faceoff']
                total_faceoffs = faceoffs_won + faceoffs_lost
                #features = []

                if counter > 300:
                    done = True

                if total_faceoffs >= 1:
                    done = True

                genome.fitness = faceoffs_won - faceoffs_lost

            logger.info("{gid} {score} {counter}",
                        gid=genome_id, score=genome.fitness, counter=counter)



def main():
    trainer = FaceoffTrainer()
    logger.info("Starting Training")
    trainer.train()

    with open('fittest.pkl', 'wb') as output:
        pickle.dump(trainer.fittest, output, 1)


if __name__ == "__main__":
    main()
