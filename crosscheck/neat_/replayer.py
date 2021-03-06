import neat
import tqdm
from typing import List, Type
from ..game_env import get_genv
from ..metascorekeeper import Metascorekeeper
from ..metascorekeeper.summer import Summer
from ..scenario import Scenario
from .. import discretizers
from typing import Callable



class Replayer:

    def __init__(self, scenario: Scenario,
                 metascorekeeper: Type[Metascorekeeper],
                 feature_vector: Callable[[dict], List[float]],
                 neat_settings_file: str,
                 discretizer: Type[discretizers.Independent] = None):
        self.scenario = scenario
        self.listeners = []
        self.metascorekeeper = metascorekeeper
        self.feature_vector = feature_vector
        self.neat_settings_file = neat_settings_file
        self.discretizer = discretizer

    def replay(self, genome: neat.DefaultGenome):
        # Create neat config
        config_filename = self.neat_settings_file

        # Setup Neat
        neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_filename)

        self._eval_genome(genome, neat_config)

    def _eval_genome(self, genome: neat.DefaultGenome, config: neat.Config):
        """
        Evaluate a single genome
        :return: The trainer's stats, as a dictionary
        """

        env = get_genv()
        if self.discretizer is not None:
            env = self.discretizer(env)

        scenario = self.scenario

        scorekeeper = scenario.scorekeeper()

        env.load_state(str(scenario.save_state))
        _ = env.reset()
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # No buttons pressed in first frame
        next_action = [0] * config.genome_config.num_outputs
        scenario.scorekeeper.env = env

        frames_since_done = 0
        total_frames = 0
        stoppage_frames = 60 * 5

        while not scorekeeper.done or frames_since_done < stoppage_frames:

            if scorekeeper.done:
                frames_since_done += 1

            # Run the next step in the simulation
            step = env.step(next_action)

            # Save the latest state
            info = step[3]
            scorekeeper.info = info

            # Determine the next action so it can be fed into the scorekeeper
            next_action = net.activate(self.feature_vector(info))
            scorekeeper.buttons_pressed = env.action_labels(next_action)

            scorekeeper.tick()

            for listener in self.listeners:
                listener(*step, {'scorekeeper': scorekeeper})
            total_frames += 1

        genome.fitness = scorekeeper.score

        msk = Summer()
        msk.add("Passthru", scorekeeper)
        return scorekeeper.stats, scorekeeper
