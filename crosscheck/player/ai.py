import neat
import pickle
import typing
from crosscheck import definitions
from crosscheck import main_train


class AiPlayer:
    """
    Represents an AI player
    """

    def __init__(self,
                 env,
                 name="FullGame-2020-02-02_16-42-2145",
                 feature_vector_name="players_and_puck_defend_up",
                 discretizer_name="2-button-bc"
                 ):
        with open(definitions.MODEL_ROOT / f"{name}.pkl", mode='rb') as f:
            genome = pickle.load(f)
        config_filename = str(definitions.MODEL_ROOT / f"{name}.ini")
        # Setup Neat
        neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_filename)
        self._net = neat.nn.recurrent.RecurrentNetwork.create(genome, neat_config)
        self._feature_vector = main_train.load_feature_vector(feature_vector_name)
        ###
        self._first = True
        self._discretizer = main_train.load_discretizer(discretizer_name)(env)
        self._next_action = []

    def step(self, info):
        next_action_2p_ai_nums = self._net.activate(self._feature_vector(info))
        self._next_action = self._discretizer.action(next_action_2p_ai_nums)
        # Swap up/down
        self.swap_up_down(self._next_action)
        self._first = False

    @classmethod
    def swap_up_down(cls, next_action):
        """
        Swap the up and down buttons. Presumably for when shoot twice direction is swapped
        :param next_action: Modifies directly
        :return: Returns next_action
        """
        tmp = next_action[4]
        next_action[4] = next_action[5]
        next_action[5] = tmp
        return next_action

    @property
    def next_action(self) -> typing.List:
        if self._first:
            # No button pressed if no inputs
            return [False for _ in range(self._discretizer.env.num_buttons)]
        else:
            return self._next_action

    def load(self):
        pass
