

class InfoWrapper:

    def __init__(self):
        # Info that comes from env
        self.info: dict = {}

    @property
    def player_w_puck(self):
        """
        Returns info about the play with the puck, including 'team' and 'position'.
        Empty dict if no possessor
        :return:
        """

        player_w_puck = {}

        for team in ('home', 'away'):
            for position in ('LW', 'C', 'RW', 'LD', 'RD', 'G'):
                possible_match = True
                for dim in ('x', 'y'):
                    player_label = 'player-{}-{}-{}'.format(team, position, dim)
                    puck_pos = 'player-w-puck-ice-{}'.format(dim)
                    if self.info[player_label] != self.info[puck_pos]:
                        possible_match = False

                if possible_match:
                    player_w_puck['team'] = team
                    player_w_puck['pos'] = position
                    return player_w_puck

        return {}

    @property
    def players_and_puck_feature(self):
        """
        :return: All of the player positions and puck position in a list. Useful as a feature vector
        """
        features = []

        for team in ('home', 'away'):
            for position in ('LW', 'C', 'RW', 'LD', 'RD', 'G'):
                for dim in ('x', 'y'):
                    player_label = 'player-{}-{}-{}'.format(team, position, dim)
                    features.append(self.info[player_label])

        for dim in ('x', 'y'):
            features.append(self.info['player-w-puck-ice-{}'.format(dim)])

        for dim in ('x', 'y'):
            features.append(self.info['puck-ice-{}'.format(dim)])

        features.append(0 if self.player_w_puck is {} else 1)

        return features
