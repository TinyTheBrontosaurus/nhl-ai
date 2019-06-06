
def get_player_w_puck(info):
    player_w_puck = {}

    for team in ('home', 'away'):
        for position in ('LW', 'C', 'RW', 'LD', 'RD', 'G'):
            possible_match = True
            for dim in ('x', 'y'):
                player_label = 'player-{}-{}-{}'.format(team, position, dim)
                puck_pos = 'player-w-puck-ice-{}'.format(dim)
                if info[player_label] != info[puck_pos]:
                    possible_match = False

            if possible_match:
                player_w_puck['team'] = team
                player_w_puck['pos'] = position
                return player_w_puck

    return {}


def players_and_puck_feature(info):
    features = []

    for team in ('home', 'away'):
        for position in ('LW', 'C', 'RW', 'LD', 'RD', 'G'):
            for dim in ('x', 'y'):
                player_label = 'player-{}-{}-{}'.format(team, position, dim)
                features.append(info[player_label])

    for dim in ('x', 'y'):
        features.append(info['player-w-puck-ice-{}'.format(dim)])

    for dim in ('x', 'y'):
        features.append(info['puck-ice-{}'.format(dim)])

    features.append(0 if get_player_w_puck(info) is {} else 1)

    return features
