from . import point_per_frame
from .base import Scorekeeper

# Hash to convert a string to a class ctor
string_to_class = {
    'point-per-frame': point_per_frame.PointPerFrame
}