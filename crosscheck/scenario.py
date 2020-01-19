import dataclasses
import retro
from .scorekeeper import Scorekeeper
from typing import Callable


@dataclasses.dataclass
class Scenario:
    name: str
    save_state: retro.State
    scorekeeper: Callable[[], Scorekeeper]
