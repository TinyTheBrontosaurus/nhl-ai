from . import winning_faceoff
from .base import Scorekeeper

# Hash to convert a string to a class ctor
string_to_class = {
    'winning-faceoff-1': winning_faceoff.FaceoffTrainer
}