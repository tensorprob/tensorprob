import numpy as np
import tensorflow as tf

from .. import config
from .base_distribution import BaseDistribution


class Normal(BaseDistribution):
    def __init__(self, mu, sigma, name=None):
        super(Normal, self).__init__(name)

        self.mu = mu
        self.sigma = sigma

        X = self
        self._logp = (
            tf.log(1 / (tf.constant(np.sqrt(2 * np.pi), dtype=config.dtype) * self.sigma)) -
            (X - self.mu)**2 / (tf.constant(2, dtype=config.dtype) * self.sigma**2)
        )

        self._cdf = lambda lim: 0.5 * tf.erfc((self.mu - lim) / (tf.constant(np.sqrt(2), config.dtype) * self.sigma))

    def logp(self):
        return self._logp

    def cdf(self, lim):
        return self._cdf(lim)
