import neat
import tqdm
import pickle
from typing import List
from loguru import logger
from crosscheck import definitions
from crosscheck.log_folder import LogFolder
from . import utils as custom_neat_utils
from ..game_env import get_genv
from ..metascorekeeper import Metascorekeeper
from typing import Callable

class Trainer:

    def __init__(self, scenarios: dict, metascorekeeper: Callable[[], Metascorekeeper]):
        self.scenarios = scenarios
        self.listeners = []
        self.metascorekeeper = metascorekeeper

    def train(self):
        # TODO: Dynamically create config
        config_filename = definitions.ROOT_FOLDER / "crosscheck" / "neat_" / \
                          "config_templates" / "config-game-scoring-1"

        # Setup Neat
        neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_filename)

        # Run tqdm and do training
        with tqdm.tqdm(smoothing=0, unit='generation') as progress_bar:
            population = neat.Population(neat_config)

            log_folder = LogFolder.folder

            population.add_reporter(custom_neat_utils.GenerationReporter(True, logger.info))
            population.add_reporter(neat.StatisticsReporter())
            population.add_reporter(custom_neat_utils.TqdmReporter(progress_bar))
            population.add_reporter(custom_neat_utils.Checkpointer(10, stream=logger.info,
                                                                   filename_prefix=log_folder / "neat-checkpoint-"))
            population.add_reporter(custom_neat_utils.SaveBestOfGeneration(log_folder / "generation-"))

            # Train
            fittest = population.run(self._eval_genomes)

            # Dump the result
            with open(log_folder / "fittest.pkl", 'wb') as f:
                pickle.dump(fittest, f, 1)

    def _eval_genomes(self, genomes: List[neat.DefaultGenome], config: neat.Config):
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

        env = get_genv()
        metascorekeeper = self.metascorekeeper

        for scenario in self.scenarios:

            name = scenario['name']
            state = scenario['state']
            scorekeeper = scenario['scorekeeper']()

            env.load_state(str(state))
            _ = env.reset()
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            # No buttons pressed in first frame
            next_action = [0] * config.genome_config.num_outputs
            scorekeeper.env = env

            while not scorekeeper.done:

                self._render()

                step = env.step(next_action)
                info = step[2]

                scorekeeper.info = info
                scorekeeper.tick()

                next_action = net.activate(self.get_inputs(info))

                for listener in self.listeners:
                    listener(*step, {'stats': scorekeeper.stats, 'score_vector': scorekeeper.score_vector})

            metascorekeeper.add(name, scorekeeper)
            self._render()

        genome.fitness = metascorekeeper.score

        return metascorekeeper.stats

    def get_inputs(self, info):
        pass

    def _render(self):
        # TODO
        pass