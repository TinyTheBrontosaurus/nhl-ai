import argparse
import sys
from crosscheck.game_env import get_genv
from crosscheck.player import human
import threading
import time
from loguru import logger
import retro
import tempfile
import pathlib
import gzip


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 reinforcement learning')

    _ = parser.parse_args(argv)

    play()


def play():

    env = get_genv()
    env.use_restricted_actions = retro.Actions.ALL
    env.reset()

    time_per_frame = 1/60

    button_state = human.ButtonState()
    button_thread = threading.Thread(target=human.maintain_button_state, args=(button_state,))
    button_thread.start()

    next_time = time.time() + time_per_frame

    env.players = 2
    frame = 0
    with tempfile.TemporaryDirectory() as state_dir_str:
        state_dir = pathlib.Path(state_dir_str)
        try:
            while True:
                # Run the next step in the simulation
                with button_state.lock:
                    next_action_dict = dict(button_state.state)

                # Convert to buttons
                next_action = [next_action_dict.get(key, 0) > 0.5 for key in env.buttons]

                # Two player?
                next_action.extend(next_action)

                _step = env.step(next_action)

                env.render()

                with gzip.open(state_dir / f"frame-{frame}.state", 'wb') as f:
                    state = env.em.get_state()
                    f.write(state)
                now = time.time()
                delay_needed = next_time - now
                if delay_needed > 0:
                    time.sleep(delay_needed)
                else:
                    logger.warning(f"Falling behind {-delay_needed:.3f}s")
                    next_time = now
                next_time += time_per_frame
                frame += 1
        finally:
            button_state.running = False


if __name__ == "__main__":
    main(sys.argv[1:])
