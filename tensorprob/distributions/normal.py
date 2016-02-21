import numpy as np
import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Normal(mu, sigma, name=None):
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = (
        tf.log(1 / (tf.constant(np.sqrt(2 * np.pi), dtype=config.dtype) * sigma)) -
        (X - mu)**2 / (tf.constant(2, dtype=config.dtype) * sigma**2)
    )

    def cdf(lim):
        return 0.5 * tf.erfc((mu - lim) / (tf.constant(np.sqrt(2), config.dtype) * sigma))

    Distribution.cdf = cdf

    return X
