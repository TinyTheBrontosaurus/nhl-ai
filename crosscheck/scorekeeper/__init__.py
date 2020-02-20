from . import point_per_frame
from . import game_scoring_1
from . import score_only
from .base import Scorekeeper

# Hash to convert a string to a class ctor
string_to_class = {
    'point-per-frame': point_per_frame.PointPerFrame,
    'game-scoring-1': game_scoring_1.GameScoring1,
    'score-only': score_only.ScoreOnly,
}