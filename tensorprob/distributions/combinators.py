import tensorflow as tf

from .. import config
from ..distribution import Distribution
from ..model import Model


@Distribution
def Mix2(f, A, B, name=None):
    # TODO(chrisburr) Check if f is bounded between 0 and 1?
    X = tf.placeholder(config.dtype, name=name)

    a_logp, a_integral, a_bounds = Model._current_model._description[A]
    b_logp, b_integral, b_bounds = Model._current_model._description[B]

    Distribution.logp = tf.log(f*tf.exp(a_logp) + (1-f)*tf.exp(b_logp))

    def integral(lower, upper):
        return f*a_integral(lower, upper) + (1-f)*b_integral(lower, upper)
    Distribution.integral = integral

    # Modify the current model to reconise that X and Y have been removed
    for dist in A, B:
        Model._current_model._silently_replace[dist] = X
        del Model._current_model._description[dist]

    return X
