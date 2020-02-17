import argparse
import sys
from crosscheck.game_env import get_genv
from crosscheck.player import human
import threading
import time
from loguru import logger
import retro
import gzip
import crosscheck.definitions
import datetime


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

    env.players = 1
    frame = 0
    state_dir = crosscheck.definitions.NEW_SAVE_STATE_FOLDER
    save_state_debounce = False
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

            save_the_state = False
            save_state_pressed = (next_action_dict.get("Z") > 0.5)

            if not save_state_debounce and save_state_pressed:
                save_the_state = True
                save_state_debounce = True
            elif not save_state_pressed:
              save_state_debounce = False

            if save_the_state:
                label = datetime.datetime.now().isoformat().replace(":", "_")
                save_file = state_dir / f"{label}.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(save_file, 'wb') as f:
                    f.write(env.em.get_state())
                logger.info(f"Saved state {str(save_file)}")

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
