
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

        if lower is None and upper is None:
            check_inside = tf.fill(tf.shape(X), True)
            lognorm = zero
        elif upper is None and isinstance(lower, BaseDistribution):
            raise ValueError("The Uniform distribution doesn't support one-sided variable boundaries")
        elif upper is None:
            check_inside = tf.greater(X, lower)
            lognorm = zero
        elif lower is None and isinstance(upper, BaseDistribution):
            raise ValueError("The Uniform distribution doesn't support one-sided variable boundaries")
        elif lower is None:
            check_inside = tf.less(X, upper)
            lognorm = zero
        else:
            check_inside = tf.logical_and(tf.greater(X, lower), tf.less(X, upper))
            lognorm = tf.log(tf.constant(1, dtype=config.dtype) / (self.upper - self.lower))

        self._logp = tf.select(check_inside, tf.fill(tf.shape(X), lognorm), tf.fill(tf.shape(X), config.dtype(-np.inf)))

    def logp(self):
        return self._logp

