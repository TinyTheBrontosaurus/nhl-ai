import typing
from .base import Scorekeeper
from ..info_utils.wrapper import InfoAccumulator, InfoWrapper

class ShootoutShooter(Scorekeeper):

    def __init__(self):
        super().__init__()

        self._accumulator = InfoAccumulator()

        self._pressed = {x: 0 for x in ['A', 'B', 'C']}

        self._juke_accumulator = 0

        self._c_ever_pressed = False
        self._c_ever_released = False
        self._c_held_for_frames = 0

        self._shot_score = 0

    def _tick(self) -> float:
        """
        Incorporate the latest frame into the total score
        :return: The total score as of this frame
        """

        # Update stats
        self._accumulator.info = self.info
        self._accumulator.accumulate()

        # End when play stops. shootout itself ends after 25s
        self._done_reasons['shoot_stopped'] = self.info['shootout-stoppage'] > 0

        # Check if haven't shot yet
        shot_this_frame = False

        if not self._c_ever_released:
            if 'C' in self.buttons_pressed:
                self._c_held_for_frames += 1
                self._c_ever_pressed = True
                # After holding it for a short time, it automatically shoots
                if self._c_held_for_frames > 30:
                    self._c_ever_released = True
                    shot_this_frame = True
            else:
                if self._c_ever_pressed:
                    # That's a shot
                    self._c_ever_released = True
                    shot_this_frame = True

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
            delta_puck_net_y = InfoWrapper.AWAY_GOAL_Y
        # No reward past the goal line
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
        # (F) Give several points for a shot
        score_vector['Press C'] = 1e4 if self._c_ever_pressed else 0
        # Give points for holding a shot, but with a time limit
        score_vector['Held C'] = 1e1 * min(self._c_held_for_frames, 2*60)
        if shot_this_frame:
            self._shot_score = juke_this_frame * 10
        score_vector['Shot'] = self._shot_score

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
            'shootout-stopped': "{}".format(self.info['shootout-stoppage']),
            'time_w_puck': ", ".join(["{} {:.1f}s".format(team, time) for team, time in self._accumulator.time_puck.items()]),
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