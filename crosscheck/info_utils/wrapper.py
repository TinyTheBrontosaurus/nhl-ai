TEAMS = ('home', 'away')
POSITIONS = ('LW', 'C', 'RW', 'LD', 'RD', 'G')
DIMS = ('x', 'y')
TIME_PER_FRAME  = 1.0 / 60


class InfoWrapper:

    # goalie x bounds: (-23, 23)
    # goalie y bounds: (245, 256)
    # end line y: ~256
    # puck in net: 272
    AWAY_GOAL_Y = 256
    GOALIE_MAX_X = 23

    def __init__(self, info: dict = None):
        """
        Grab derived info directly from an individual info object. Basically f(info)
        """
        # Info that comes from env
        if info is None:
            info = {}
        self.info: dict = info

    @property
    def player_w_puck(self):
        """
        Returns info about the play with the puck, including 'team' and 'position'.
        Empty dict if no possessor
        :return:
        """
        player_w_puck = {}
        diff = {}
        for team in TEAMS:
            for position in POSITIONS:
                total_diff = 0
                for dim in DIMS:
                    player_label = 'player-{}-{}-{}'.format(team, position, dim)
                    puck_pos = 'player-w-puck-ice-{}'.format(dim)

                    # Found empirically that the pixel can be off by two
                    diff[player_label] = self.info[player_label] - self.info[puck_pos]
                    total_diff += abs(diff[player_label])

                if total_diff <= 2:
                    player_w_puck['team'] = team
                    player_w_puck['pos'] = position
                    return player_w_puck

        return {}

    @property
    def puck_adjusted_away_goalie_x(self):
        return min(max(self.info['puck-ice-x'], -self.GOALIE_MAX_X), self.GOALIE_MAX_X)

    @property
    def delta_puck_away_goalie_x(self):
        return abs(self.puck_adjusted_away_goalie_x - self.info['player-away-G-x'])

    @property
    def delta_puck_away_net_y(self):
       return self.AWAY_GOAL_Y - self.info['puck-ice-y']

    @property
    def delta_slot_net_y(self):
        """
        How far the slot is from the net
        """
        return 100


class InfoAccumulator:


    def __init__(self):
        """
        Grab derived info from the history of info objects. Basically f(info_n, info_{n-1})
        """

        self.wrapper = InfoWrapper()
        self.time_puck = {
            'home': 0.0,
            None: 0.0,
            'away': 0.0,
        }

        self.pass_completions = {
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

        self._max_puck_y = 0
        self._max_shooter_y = 0

        self._frame_counter = 0
        self._last_tick_frame = 0
        self._last_time = None

    @property
    def info(self):
        return self.wrapper.info

    @info.setter
    def info(self, info):
        self.wrapper.info = info

    def _mark_received_puck(self, player_w_puck):
        # Make sure this isn't an initial possession
        if self._last_possessor:
            if player_w_puck['team'] == self._last_possessor['team']:
                # via pass
                self.pass_completions[player_w_puck['team']] += 1
            else:
                # via turnover
                self.steal_count[player_w_puck['team']] += 1

        self._possession_history['team'].append(player_w_puck['team'])
        self._possession_history['pos'].append(player_w_puck['pos'])

    def _mark_caught_own_pass(self, player_w_puck):
        self.pass_attempts[player_w_puck['team']] -= 1

    def _mark_lost_puck(self):
        # Note: Very loose definition of pass
        self.pass_attempts[self._last_frame_player_w_puck['team']] += 1

    def accumulate(self):

        player_w_puck = self.wrapper.player_w_puck

        # Accumulate time with the puck
        self.time_puck[player_w_puck.get('team')] += TIME_PER_FRAME

        # Someone just got the puck
        if player_w_puck and not self._last_frame_player_w_puck:
            if self._last_possessor and all([player_w_puck[key] == self._last_possessor[key] for key in player_w_puck.keys()]):
                self._mark_caught_own_pass(player_w_puck)
            else:
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

        # Keep track how deep the puck has gone
        self._max_puck_y = max(self.info['puck-ice-y'], self._max_puck_y)

        # Keep track of how deep the puck has gone when it's possessed
        if player_w_puck['team'] == 'home':
            self._max_shooter_y = max(self.info['puck-ice-y'], self._max_shooter_y)

        # Keep track of ticks
        self._frame_counter += 1
        if self._last_time != self.info['time']:
            self._last_tick_frame = self._frame_counter
            self._last_time = self.info['time']

    @property
    def consecutive_passes(self):
        """
        Return stats per team showing the number of consecutive passes (in a list) and the number of
        those passes that are unique (that is, up to 6 different players in a row)
        """
        consecutive = {'home': [0], 'away': [0]}
        unique = {'home': [0], 'away': [0]}

        if len(self._possession_history['team']):

            pvs_team = self._possession_history['team'][0]
            unique_local = [self._possession_history['pos'][0]]

            for cur_team, cur_pos in zip(self._possession_history['team'][1:], self._possession_history['pos'][1:]):
                if cur_team == pvs_team:
                    consecutive[cur_team][-1] += 1

                    # Track consecutive unique passes. As soon as the unique streak is broken, the entire set is reset
                    if cur_pos in unique_local:
                        unique_local = [cur_pos]
                        unique[cur_team].append(0)
                    else:
                        unique_local.append(cur_pos)
                        unique[cur_team][-1] += 1

                        # If everyone has touched the puck (all 6 players) then reset the unique list
                        if len(unique_local) >= 6:
                            unique_local = []
                else:
                    consecutive[cur_team].append(0)
                    unique[cur_team].append(0)
                    unique_local = []


                pvs_team = cur_team

        return {'consecutive': consecutive, 'unique': unique}

    @property
    def max_puck_y(self):
        return self._max_puck_y

    @property
    def max_shooter_y(self):
        return self._max_shooter_y

    @property
    def has_play_stopped_after_game_start(self):
        """
        True if play has stopped, but does not include before the first puck drop
        """
        if self.info['time'] < 600:
            if self._frame_counter - self._last_tick_frame > 120:
                return True
        return False