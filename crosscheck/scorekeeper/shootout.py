import typing
from .base import Scorekeeper
from ..info_utils.wrapper import InfoAccumulator, InfoWrapper

class Shootout(Scorekeeper):

    def __init__(self):
        super().__init__()

        self._accumulator = InfoAccumulator()

        self._pressed = {x: 0 for x in ['A', 'B', 'C']}

        self._juke_accumulator = 0

    def _tick(self) -> float:
        """
        Incorporate the latest frame into the total score
        :return: The total score as of this frame
        """

        # Update stats
        self._accumulator.info = self.info
        self._accumulator.accumulate()

        # End when play stops. shootout itself ends after 25s
        self._done_reasons['shoot_stopped'] = self.info['shootout-stoppage'] and True


        # Rewards structure
        # The intent it to reward gradually in the following order:
        #  * B) moving the puck towards the net
        #  * D) shooting the puck
        #  * E) moving the puck to a high probability location
        #  * G) shooting the puck from a high probability location
        #  * G) shooting the puck fast
        #  * H) scoring

        self._score_vector = {}
        score_vector = self._score_vector
        # (B) Total max: ~3000
        # (B) Bonus points for sending the puck deep
        # Max (~1000)
        score_vector['deep-puck-dump'] = self._accumulator.max_puck_y * 4
        # (B) Bonus points for player taking the puck deep
        # Max (~2000)
        score_vector['deep-puck-carry'] = self._accumulator.max_shooter_y * 8

        # (D) Total max: ~50k (assuming 5 shots is a realistic max)
        # Reward shots on goal. Allow grinding
        score_vector['home-shots'] = self.info['home-shots'] * 1e4

        # (D.2) Give 100k points for the first shot. This to make sure they shoot at least once
        score_vector['any-shot'] = 1e5 if self.info['home-shots'] > 0 else 0

        # (E) Total max: ~50k (with multiplier of 0.01)
        # If behind the net, give the same reward as the away side of center ice,
        # to avoid behind-the-net grinding
        delta_puck_net_y = self._accumulator.wrapper.delta_puck_away_net_y
        if delta_puck_net_y <= 2:
            delta_puck_net_y = 225
        # No reward past the red line
        distance_multiplier = max(InfoWrapper.AWAY_GOAL_Y - delta_puck_net_y, 0)

        if self._accumulator.wrapper.player_w_puck.get('team') == 'home':
            juke_this_frame = self._accumulator.wrapper.delta_puck_away_goalie_x * distance_multiplier
        else:
            juke_this_frame = 0
        # Theoretical max of accumulator is 60s * 60frames * 50 x-pixels * 250 y-pixels == 45M
        # Realistic (human) max of accumulator is a 1Hz sine wave towards the goalie,
        # average y-distance of 100, average x-distance of 1 * 60fps, 60s == 3,600,000
        self._juke_accumulator += juke_this_frame

        # Reward all jukes
        score_vector['juke'] = self._juke_accumulator * 0.05

        # TODO (F and G) both required a shot detector

        # (H) Total max: 500k (to leave room for F and G)
        score_vector['home-goals'] = self.info['home-goals'] * 5e5

        score = sum(score_vector.values())

        # Accumulate button presses
        if 'A' in self.buttons_pressed:
            self._pressed['A'] += 1
        if 'B' in self.buttons_pressed:
            self._pressed['B'] += 1
        if 'C' in self.buttons_pressed:
            self._pressed['C'] += 1

        # Save the score vector
        self._score_vector = score_vector

        # Stats to track
        self._stats = {
            'time_w_puck': ", ".join(["{} {:.1f}s".format(team, time) for team, time in self._accumulator.time_puck.items()]),
            'buttons': self._pressed,
        }

        return score


        if info['home-goals']:
            score += 100000
            # Try to solve as quickly as possible
            score += info['time-shootout'] * 1000

        # Stop early once stoppage occurs
        if info['shootout-stoppage']:
            if self.short_circuit:
                self._done = True
            self._pending_stoppage = True
        elif self._pending_stoppage:
            self._done = True

        features = [
            shooter_x,
            shooter_y,
            goalie_x,
            goalie_y,
            puck_x,
            puck_y
        ]
        self._next_action = self.net.activate(features)

        self._stats = {"score": score}

        return score


    @classmethod
    def fitness_threshold(cls) -> float:
        """
        Accessor fitness threshold (the score over
        which to stop training)
        """
        return 7e5
