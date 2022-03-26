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

        self._max_puck_y = max(puck_y, self._max_puck_y)
        self._max_shooter_y = max(puck_y, self._max_shooter_y)

        # goalie x bounds: (-23, 23)
        # goalie y bounds: (245, 256)
        # end line y: ~256
        # puck in net: 272
        #
        # reward:
        # * difference between puck and goalie, bounded. X preferred. bonus as Y is different
        # * x speed of puck?
        # * y position of player, bounded. encourage moving towards goalie
        # * shot location? want to discourage wasted shot
        # * time elapsed? again, want to dicourage wasted shot

        puck_adjusted_goalie_x = min(max(puck_x, -23), 23)

        delta_puck_goalie_x = abs(puck_adjusted_goalie_x - goalie_x)

        # Lower numbers are better. but if behind the next (256) treat it the same as far from the net
        delta_puck_net_y = self.GOAL_Y - puck_y
        # If very close to the net (or behind), give the same credit as if above the slot
        if delta_puck_net_y <= 2:
            delta_puck_net_y = 100

        distance_multiplier = max(self.GOAL_Y - delta_puck_net_y, 0)

        juke_reward = delta_puck_goalie_x * distance_multiplier * 0.001

        # Reward all jukes
        self._score_acc += juke_reward

        score = self._score_acc

        score += max(self._max_puck_y, self.GOAL_Y)
        score += max(self._max_shooter_y, self.GOAL_Y)

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



        # Update stats
        self._accumulator.info = self.info
        self._accumulator.accumulate()

        # End if a minute has passed,
        self._done_reasons['timeout'] = self.info['time'] <= 540
        # away scores a goal (fail)--which is most likely an own goal,
        self._done_reasons['away_score'] = self.info['away-goals'] > 0
        # home scores a goal (success!),
        self._done_reasons['home_score'] = self.info['home-goals'] > 0
        # when the away team has the puck for  too long
        self._done_reasons['away_has_puck'] = self._accumulator.time_puck['away'] > 1
        # when play stops
        self._done_reasons['play_stopped'] = self._accumulator.has_play_stopped_after_game_start


        # Rewards structure
        # The intent it to reward gradually in the following order:
        #  * A) winning the faceoff, and keeping the puck
        #  * B) moving the puck towards the net
        #  * C) successful passing / not losing the puck
        #  * D) shooting the puck
        #  * E) moving the puck to a high probability location
        #  * G) shooting the puck from a high probability location
        #  * G) shooting the puck fast
        #  * H) scoring

        self._score_vector = {}
        score_vector = self._score_vector
        # (A) Total Max: 360
        # (A) Some points for every second with the puck.
        # Max given 60x accumulation (300)
        score_vector['possession-home'] = self._accumulator.time_puck['home'] * 5
        # (A) Very few points when no one has the puck (accounts for play stoppages too)
        # Max given 60x accumulation (60)
        score_vector['possession-None'] = self._accumulator.time_puck[None] * 1
        # (A) Penalty when the other team has it (capped due to ending the actual simulation after
        # not too long)
        # Max given 60x accumulation  (-180)
        score_vector['possession-away'] = self._accumulator.time_puck['away'] * -3

        # (B) Total max: ~3000
        # (B) Bonus points for sending the puck deep
        # Max (~1000)
        score_vector['deep-puck-dump'] = self._accumulator.max_puck_y * 4
        # (B) Bonus points for player taking the puck deep
        # Max (~2000)
        score_vector['deep-puck-carry'] = self._accumulator.max_shooter_y * 8

        # (C) Total max: 5,000
        # Reward passes, but don't allow grinding
        # Max 1x (2,500)
        MAX_PASS_COUNT_REWARD = 5
        score_vector['pass-att'] = min(self._accumulator.pass_attempts['home'], MAX_PASS_COUNT_REWARD) * 500
        # Reward completion, but keep adjusting for cmp percentage
        # Max 1x (2,500)
        score_vector['pass-com'] = min(self._accumulator.pass_completions['home'], MAX_PASS_COUNT_REWARD) * cmp_pct * 500

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


        # Fix when the opponent wins the faceoff but gets points for taking
        # the puck up ice
        # If the faceoff was lost, then don't count any points
        if self._done_reasons.get('lost_faceoff'):
            for key in self._score_vector:
                score_vector[key] = 0

        score = sum(score_vector.values())

        # Calculate commands based on features
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
            'pass cmp/att': "{}/{} ({:.0f}%) shots={}".format(self._accumulator.pass_completions['home'],
                                           self._accumulator.pass_attempts['home'],
                                                   cmp_pct * 100, self.info['home-shots']),
            'buttons': self._pressed,
        }

        return score

    @classmethod
    def fitness_threshold(cls) -> float:
        """
        Accessor fitness threshold (the score over
        which to stop training)
        """
        return 7e5
