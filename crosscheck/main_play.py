import argparse
import sys
from crosscheck.discretizers import Genesis6ButtonWithStart
from crosscheck.game_env import get_genv


def main(argv):
    parser = argparse.ArgumentParser(description='Cross-check: NHL \'94 reinforcement learning')

    _ = parser.parse_args(argv)

    play()


def play():

    env = get_genv()
    env = Genesis6ButtonWithStart(env)

    next_action

    while True:
        # Run the next step in the simulation
        step = env.step(next_action)


if __name__ == "__main__":
    main(sys.argv[1:])