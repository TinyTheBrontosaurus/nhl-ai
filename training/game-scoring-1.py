import sys
from training import runner
from training import discretizers
import neat
import typing
from training.info_utils import InfoAccumulator, InfoWrapper


class GameScoring1Trainer(runner.Trainer):

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, short_circuit: bool):
        super().__init__(genome, config, short_circuit)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        self._stats = {}

        self._done = False

        self._next_action = None

        self._accumulator = InfoAccumulator()

        self._pressed = {x: 0 for x in ['A', 'B', 'C']}

        self._juke_accumulator = 0

    @classmethod
    def get_config_filename(cls) -> str:
        """
        Accessor for config filename
        """
        return 'config-game-scoring-1'

    @classmethod
    def get_scenario_string(cls) -> str:
        """
        Accessor for the scenario to use
        """
        return 'ChiAtBuf-Faceoff'

    @classmethod
    def discretizer_class(cls) -> typing.Callable[[], discretizers.Genesis3ButtonDiscretizer]:
        return discretizers.Genesis3ButtonDiscretizer

    @property
    def next_action(self) -> list:
        return self._next_action

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def done(self) -> bool:
        return self._done

    def tick(self, ob, rew, done, info, env) -> float:

        # Update stats
        self._accumulator.info = info
        self._accumulator.accumulate()

        att = self._accumulator.pass_attempts['home']
        cmp = self._accumulator.pass_completions['home']

        cmp_pct = 0.0

        if att > 1:
            cmp_pct = cmp / (att - 1)  # The latest one can't be counted since it might still be happening
            threshold = 0.75  # 3/4 must be completed

            # End if passing is a mess
            if cmp_pct < threshold:
                self._done = True
        elif self._accumulator.wrapper.player_w_puck.get('team') == 'away':
            # End if the other team gets the puck very early
            self._done = True


        # End if a minute has passed,
        if ( info['time'] <= 540
                # away scores a goal (fail)--which is most likely an own goal,
                or info['away-goals'] > 0
                # home scores a goal (success!), or
                or info['home-goals'] > 0
                # when the away team has the puck for  too long
                or self._accumulator.time_puck['away'] > 5
             ):
            self._done = True

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

        score_vector = []
        # (A) Total Max: 360
        # (A) Some points for every second with the puck.
        # Max given 60x accumulation (300)
        score_vector.append(self._accumulator.time_puck['home'] * 5)
        # (A) Very few points when no one has the puck (accounts for play stoppages too)
        # Max given 60x accumulation (60)
        score_vector.append(self._accumulator.time_puck[None] * 1)
        # (A) Penalty when the other team has it (capped due to ending the actual simulation after
        # not too long)
        # Max given 60x accumulation  (-180)
        score_vector.append(self._accumulator.time_puck['away'] * -3)

        # (B) Total max: ~3000
        # (B) Bonus points for sending the puck deep
        # Max (~1000)
        score_vector.append(self._accumulator.max_puck_y * 4)
        # (B) Bonus points for player taking the puck deep
        # Max (~2000)
        score_vector.append(self._accumulator.max_puck_y * 8)

        # (C) Total max: 5000
        # Reward passes, but don't allow grinding
        # Max 1x (2500)
        MAX_PASS_COUNT_REWARD = 5
        score_vector.append(min(self._accumulator.pass_attempts['home'], MAX_PASS_COUNT_REWARD) * 500)
        # Reward completion, but keep adjusting for cmp percentage
        # Max 1x (2500)
        score_vector.append(min(self._accumulator.pass_completions['home'], MAX_PASS_COUNT_REWARD) * cmp_pct * 500)

        # (D) Total max: ~50,000 (assuming 5 shots is a realistic max)
        # Reward shots on goal. Allow grinding
        score_vector.append(info['home-shots'] * 10000)

        # (E) Total max: ~500k
        # If behind the net, give the same reward as the away side of center ice,
        # to avoid behind-the-net grinding
        delta_puck_net_y = self._accumulator.wrapper.delta_puck_net_y
        if delta_puck_net_y <= 2:
            delta_puck_net_y = 225
        # No reward past the red line
        distance_multiplier = max(InfoWrapper.AWAY_GOAL_Y - delta_puck_net_y, 0)

        juke_this_frame = self._accumulator.wrapper.delta_puck_goalie_x * distance_multiplier
        # Theoretical max of accumulator is 60s * 60frames * 50 x-pixels * 250 y-pixels == 45M
        # Realistic (human) max of accumulator is a 1Hz sine wave towards the goalie,
        # average y-distance of 100, average x-distance of 1 * 60fps, 60s == 3,600,000
        self._juke_accumulator += juke_this_frame

        # Reward all jukes
        score_vector.append(self._juke_accumulator * 0.1)

        # TODO (F and G) both required a shot detector

        # (H) Total max: 10M (to leave room for F and G)
        score_vector.append(info['home-goals'] * 1e7)

        score = sum(score_vector)

        # Calculate commands based on features
        self._next_action = self.net.activate(self._accumulator.wrapper.players_and_puck_feature)
        buttons_pressed = env.action_labels(self._next_action)
        if 'A' in buttons_pressed:
            self._pressed['A'] += 1
        if 'B' in buttons_pressed:
            self._pressed['B'] += 1
        if 'C' in buttons_pressed:
            self._pressed['C'] += 1

        # Stats to track
        self._stats = {
            'score': score,
            'time_w_puck': ", ".join(["{} {.1}s".format(team, time) for team, time in self._accumulator.time_puck.items()]),
            'pass cmp/att': "{}/{} (:.0f}%".format(self._accumulator.pass_completions['home'],
                                           self._accumulator.pass_attempts['home'], cmp_pct * 100),
            'buttons': self._pressed,

        }

        return score


if __name__ == "__main__":
    runner.main(sys.argv[1:], GameScoring1Trainer)
