import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Polynomial(coefficients, name=None):
    assert len(coefficients) >= 2

    for coeff in coefficients:
        assert isinstance(coeff, tf.Tensor)

    X = tf.placeholder(config.dtype, name=name)

    pdf = []
    for i, coeff in enumerate(coefficients):
        pdf.append(coeff * X**tf.constant(i, dtype=config.dtype))
    Distribution.logp = tf.log(tf.add_n(pdf))

    def integrate(lower, upper):
        result = []
        for i, coeff in enumerate(coefficients, start=1):
            order = tf.constant(i, dtype=config.dtype)
            result.append(coeff/order * (upper**order - lower**order))
        return tf.add_n(result)

    Distribution.integral = integrate

    return X
