

class InfoWrapper:

    def __init__(self):
        """
        Grab derived info from an info object
        """
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

TIME_PER_FRAME  = 1.0 / 60

class InfoAccumulator:


    def __init__(self):
        self._wrapper = InfoWrapper()
        self.time_puck = {
            'home': 0.0,
            None: 0.0,
            'away': 0.0,
        }

        self.pass_count = {
            'home': 0,
            'away': 0,
        }

        self.steal_count = {
            'home': 0,
            'away': 0,
        }

        self.pass_attempts = {
            'home': 0,
            'away': 0,
        }

        # Indicates who had the puck in the last frame
        self._last_frame_player_w_puck = {}

        # Possessor is guaranteed to be the last player that held it. Will only be null at the start
        self._last_possessor = {}

        # Chronological order, earliest at the start
        self._possession_history = {
            'team': [],
            'pos': [],
        }

    @property
    def info(self):
        return self._wrapper.info

    @info.setter
    def info(self, info):
        self._wrapper.info = info

    def _mark_received_puck(self, player_w_puck):
        # Make sure this isn't an initial possession
        if self._last_possessor:
            if player_w_puck['team'] == self._last_possessor['team']:
                # via pass
                self.pass_count[player_w_puck['team']] += 1
            else:
                # via turnover
                self.steal_count[player_w_puck['team']] += 1

        self._possession_history['team'].append(player_w_puck['team'])
        self._possession_history['pos'].append(player_w_puck['pos'])

    def _mark_lost_puck(self):
        # Note: Very loose definition of pass
        self.pass_attempts[self._last_frame_player_w_puck['team']] += 1

    def accumulate(self):

        player_w_puck = self._wrapper.player_w_puck

        # Accumulate time with the puck
        self.time_puck[player_w_puck.get('team')] += TIME_PER_FRAME

        # Someone just got the puck
        if player_w_puck and not self._last_frame_player_w_puck:
            self._mark_received_puck(player_w_puck)

        # Someone just lost the puck
        elif not player_w_puck and self._last_frame_player_w_puck:
            self._mark_lost_puck()

        # Someone stole the puck directly
        elif any([player_w_puck[key] != self._last_frame_player_w_puck[key] for key in player_w_puck.keys()]):
            self._mark_received_puck(player_w_puck)
            self._mark_lost_puck()

        # Puck hasn't changed possession (either same player has it or no player has it)
        else:
            pass


        ## Cleanup

        # If a player possesses the puck, save that info for next frame
        if player_w_puck:
            self._last_possessor = player_w_puck

        # Save the player w/ puck for next frame
        self._last_frame_player_w_puck = player_w_puck

    @property
    def consecutive_passes(self):
        """
        Return stats per team showing the number of consecutive passes (in a list) and the number of
        those passes that are unique (that is, up to 6 different players in a row)
s        """
        consecutive = {'home': [], 'away': []}
        unique = {'home': [], 'away': []}
        unique_local = []

        if len(self._possession_history):

            pvs = self._possession_history['team']
            for cur in self._possession_history[1:]:
                if cur == pvs:
                    consecutive[cur][-1] += 1

                    # Track consecutive unique passes
                    if self._possession_history['pos'] not in unique_local:
                        unique_local.append(self._possession_history['pos'])
                        unique[cur][-1] += 1

                        # If everyone has touched the puck (all 6 players) then reset the unique list
                        if len(unique) >= 6:
                            unique_local = []
                else:
                    consecutive[cur].append(0)
                    unique[cur].append(0)
                    unique_local = []


                pvs = cur

        return {'consecutive': consecutive, 'unique': unique}
