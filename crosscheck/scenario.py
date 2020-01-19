import dataclasses
import retro
from .scorekeeper import Scorekeeper
from typing import Type


@dataclasses.dataclass
class Scenario:
    name: str
    save_state: retro.State
    scorekeeper: Type[Scorekeeper]
