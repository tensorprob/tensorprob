
import numpy as np
import tensorflow as tf

from .. import config
from .base_distribution import BaseDistribution

class Uniform(BaseDistribution):
    def __init__(self, lower=None, upper=None, name=None):
        super(Uniform, self).__init__(name)

        self.lower = lower
        self.upper = upper

        X = self

        zero = tf.constant(0, dtype=config.dtype)
        one = tf.constant(1, dtype=config.dtype)

        if lower is None and upper is None:
            check_inside = tf.fill(tf.shape(X), True)
            norm = one
        elif upper is None:
            check_inside = tf.greater(X, lower)
            norm = one
        elif lower is None:
            check_inside = tf.less(X, upper)
            norm = one
        else:
            check_inside = tf.logical_and(tf.greater(X, lower), tf.less(X < upper))
            norm = 1 / (self.upper - self.lower)

        self._logp = tf.select(check_inside, tf.fill(tf.shape(X), zero), tf.fill(tf.shape(X), config.dtype(-np.inf)))

    def logp(self):
        return self._logp

