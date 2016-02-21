import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Polynomial(coefficients, name=None):
    for coeff in coefficients:
        assert isinstance(coeff, tf.Tensor)

    X = tf.placeholder(config.dtype, name=name)

    pdf = coefficients[0]
    for i, coeff in enumerate(coefficients[1:], start=1):
        pdf += coeff * X**i
    Distribution.logp = tf.log(pdf)

    def cdf(lim):
        result = 0
        for i, coeff in enumerate(coefficients[1:]):
            result += coefficients * lim**i
        return result

    Distribution.integral = lambda lower, upper: cdf(upper) - cdf(lower)

    return X
