from workflow import functional
from workflow import ignite
from workflow import torch

from .figure_to_numpy import figure_to_numpy
from .split_new_data import split_new_data

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution('ml-workflow').version
except DistributionNotFound:
    pass
