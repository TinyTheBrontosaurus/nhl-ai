from typing import Type
import pathlib
from dataclasses import dataclass
from crosscheck.scorekeeper import Scorekeeper
from crosscheck.real_time_game import RealTimeGame
from crosscheck.practice.menu import MenuHandler
from crosscheck.player.rendering import SimpleImageViewer
from crosscheck.player.human import ButtonState
from loguru import logger
from crosscheck.db.human import add_round
import datetime


@dataclass
class Minigame:

    def __init__(self, pkey: str,
        scenario: pathlib.Path,
        timeout: float,
        scorekeeper_type: Type[Scorekeeper],
        iterations: int,
        button_state: ButtonState,
        viewer: SimpleImageViewer,
        menu: MenuHandler):
        self.pkey = pkey
        self.scenario = scenario
        self.timeout = timeout
        self.scorekeeper_type = scorekeeper_type
        self.iterations = iterations
        self.button_state = button_state
        self.viewer = viewer
        self.menu = menu
        self.total_score = 0
        self.max_possible_score = 0


    def play(self):
        self.total_score = 0
        gametime = datetime.datetime.now()
        for attempt in range(self.iterations):
            scorekeeper = self.scorekeeper_type()
            player = RealTimeGame(self.button_state, self.scenario, self.viewer, self.menu, scorekeeper, self.timeout,
                                  True, exit_on_y=True)
            player.play()
            add_round(self.scenario, scorekeeper.score > 0.5, player.button_presses, gametime=gametime)
            self.total_score += scorekeeper.score
            self.max_possible_score += scorekeeper.fitness_threshold()
            logger.info(f"Score {self.total_score}/{self.max_possible_score}")

