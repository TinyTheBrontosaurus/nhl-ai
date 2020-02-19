from typing import Type
import pathlib
from dataclasses import dataclass
from crosscheck.scorekeeper import Scorekeeper
from crosscheck.real_time_game import RealTimeGame
from crosscheck.practice.menu import MenuHandler
from crosscheck.player.rendering import SimpleImageViewer
from crosscheck.player.human import ButtonState

@dataclass
class Minigame:

    #def __init__(self, pkey, scenario, timeout, scorekeeper_type, iterations, button_state, viewer, menu):
    pkey: str
    scenario: pathlib.Path
    timeout: float
    scorekeeper_type: Type[Scorekeeper]
    iterations: int
    button_state: ButtonState
    viewer: SimpleImageViewer
    menu: MenuHandler


    def play(self):
        total_score = 0
        for attempt in range(self.iterations):
            scorekeeper = self.scorekeeper_type()
            player = RealTimeGame(self.button_state, self.scenario, self.viewer, self.menu, scorekeeper)
            player.play()
            total_score += scorekeeper.score


