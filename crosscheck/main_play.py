import argparse
import sys
from crosscheck.discretizers import Genesis6ButtonWithStart
from crosscheck.game_env import get_genv
from crosscheck.player import human
import threading


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 reinforcement learning')

    _ = parser.parse_args(argv)

    play()


def play():

    env = get_genv()
    env = Genesis6ButtonWithStart(env)

    button_state = human.ButtonState()
    button_thread = threading.Thread(target=human.maintain_button_state, args=(button_state,))
    button_thread.start()

    while True:
        # Run the next step in the simulation
        with button_state.lock:
            next_action = dict(button_state.state)
        _step = env.step(next_action)
        env.render()


if __name__ == "__main__":
    main(sys.argv[1:])
