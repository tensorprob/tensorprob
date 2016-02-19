import numpy as np
import tensorflow as tf

from .. import config
from .base_distribution import BaseDistribution


class Normal(BaseDistribution):
    def __init__(self, mu, sigma, name=None):
        self.mu = mu
        self.sigma = sigma

        super(Normal, self).__init__(name)

    def log_pdf(self, X):
        # log(1/[sqrt(2*pi)*sigma]) - (x-mu)^2/(2*sigma^2)
        return (
            tf.log(1 / (tf.constant(np.sqrt(2*np.pi), dtype=config.dtype)*self.sigma)) -
            tf.pow(X-self.mu, 2) / (tf.constant(2, dtype=config.dtype)*tf.pow(self.sigma, 2))
        )

    def cdf(self, lim):
        # 0.5 * erfc([mu-x] / [sqrt(2)*sigma])
        return 0.5 * tf.erfc(
            (self.mu-lim) / (tf.constant(np.sqrt(2), config.dtype)*self.sigma)
        )
