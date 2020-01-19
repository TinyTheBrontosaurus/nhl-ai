from typing import List
from .wrapper import TEAMS, POSITIONS, DIMS, InfoWrapper


def players_and_puck(info: dict) -> List[float]:
    """
    :return: All of the player positions and puck position in a list.
    Useful as a feature vector
    """
    features = []

    wrapper = InfoWrapper(info)

    for team in TEAMS:
        for position in POSITIONS:
            for dim in DIMS:
                player_label = 'player-{}-{}-{}'.format(team, position, dim)
                features.append(wrapper.info[player_label])

    for dim in DIMS:
        features.append(wrapper.info['player-w-puck-ice-{}'.format(dim)])

    for dim in DIMS:
        features.append(wrapper.info['puck-ice-{}'.format(dim)])

    features.append(0 if wrapper.player_w_puck is {} else 1)

    return features

# Hash to convert a string to a class ctor
string_to_class = {
    'players_and_puck': players_and_puck
}
