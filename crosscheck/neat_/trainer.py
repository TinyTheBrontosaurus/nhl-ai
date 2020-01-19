import neat
import tqdm
import pickle
import pathlib
import configparser
from typing import List
from loguru import logger
from crosscheck import definitions
from crosscheck.log_folder import LogFolder
from . import utils as custom_neat_utils
from ..game_env import get_genv
from ..metascorekeeper import Metascorekeeper
from ..scenario import Scenario
from typing import Callable

class Trainer:

    def __init__(self, scenarios: List[Scenario],
                 metascorekeeper: Callable[[], Metascorekeeper],
                 feature_vector: Callable[[dict], List[float]],
                 neat_settings: dict = None):
        self.scenarios = scenarios
        self.listeners = []
        self.metascorekeeper = metascorekeeper
        self.feature_vector = feature_vector
        if neat_settings is None:
            neat_settings = {}
        self.neat_settings = neat_settings

    def setup_neat_config(self) -> pathlib.Path:
        """
        Dynamically create config from a template, and store it in the log folder
        :return: Path to new config
        """
        log_folder = LogFolder.folder

        # Read template
        parser = configparser.ConfigParser()
        template_config_filename = definitions.ROOT_FOLDER / "crosscheck" / "neat_" / \
                                   "config_templates" / "config-game-scoring-1"
        parser.read(template_config_filename)

        # Set
        parser["NEAT"]["fitness_threshold"] = "250000"

        # Apply user config last (top priority)
        for key, subkeys in self.neat_settings.items():
            for subkey, value in subkeys.items():
                if key not in parser:
                    raise KeyError(f"Unrecognized config key for neat config: {key}")
                if subkey not in parser[key]:
                    raise KeyError(f"Unrecognized config subkey for neat config: {key}/{subkey}")

                parser[key][subkey] = str(value)

        # Write
        config_filename = log_folder / "neat_config.ini"
        with open(config_filename, 'w') as f:
            parser.write(f)

        return config_filename

    def train(self):
        # Create neat config
        config_filename = self.setup_neat_config()

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
            checkpoint_folder = log_folder / "checkpoints"
            checkpoint_folder.mkdir(parents=False, exist_ok=False)
            population.add_reporter(custom_neat_utils.Checkpointer(10, stream=logger.info,
                                                                   filename_prefix=checkpoint_folder / "neat-checkpoint-"))
            generations_folder = (log_folder / "generations")
            generations_folder.mkdir(parents=False, exist_ok=False)
            population.add_reporter(custom_neat_utils.SaveBestOfGeneration(generations_folder / "generation-"))

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
        metascorekeeper = self.metascorekeeper()

        for scenario in self.scenarios:

            scorekeeper = scenario.scorekeeper()

            env.load_state(str(scenario.save_state))
            _ = env.reset()
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            # No buttons pressed in first frame
            next_action = [0] * config.genome_config.num_outputs
            scenario.scorekeeper.env = env

            while not scorekeeper.done:

                self._render()

                step = env.step(next_action)
                info = step[3]

                scorekeeper.info = info
                scorekeeper.tick()

                next_action = net.activate(self.feature_vector(info))

                for listener in self.listeners:
                    listener(*step, {'stats': scorekeeper.stats, 'score_vector': scorekeeper.score_vector})

            metascorekeeper.add(scenario.name, scorekeeper)
            self._render()

        genome.fitness = metascorekeeper.score

        return metascorekeeper.stats

    def _render(self):
        # TODO
        pass