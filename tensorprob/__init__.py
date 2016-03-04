__version__ = '0.0.0'

import logging
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.WARN)

import colorlog
from colorlog import ColoredFormatter

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'blue',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

logger = colorlog.getLogger('tensorprob')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

from . import config
from . import utilities
from . import distributions
from .distribution import Distribution, DistributionError
from .model import Model, ModelError
from .parameter import Parameter
from .stats import fisher
from .distributions import *
from .optimizers import *
from .samplers import *
from .utilities import Region
