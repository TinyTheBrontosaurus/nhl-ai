from . import summer
from . import mean
from . import median
from.base import Metascorekeeper

# Hash to convert a string to a class ctor
string_to_class = {
    'summer': summer.Summer,
    'mean': mean.Mean,
    'median': median.Median,
}