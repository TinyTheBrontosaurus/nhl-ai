import argparse
import sys
from crosscheck.game_env import get_genv
from crosscheck.player import human
import threading
import time
from loguru import logger


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 reinforcement learning')

    _ = parser.parse_args(argv)

    play()


def play():

    env = get_genv()
    env.reset()

    time_per_frame = 1/60

    button_state = human.ButtonState()
    button_thread = threading.Thread(target=human.maintain_button_state, args=(button_state,))
    button_thread.start()

    next_time = time.time() + time_per_frame

    while True:
        # Run the next step in the simulation
        with button_state.lock:
            next_action_dict = dict(button_state.state)

        # Convert to buttons
        next_action = [next_action_dict.get(key, 0) for key in env.buttons]

        _step = env.step(next_action)
        env.render()
        now = time.time()
        delay_needed = next_time - now
        if delay_needed > 0:
            time.sleep(delay_needed)
        else:
            logger.warning(f"Falling behind {-delay_needed:.3f}s")
        next_time += time_per_frame


if __name__ == "__main__":
    main(sys.argv[1:])
