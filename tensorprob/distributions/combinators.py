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

    # Modify the current model to recognize that X and Y have been removed
    for dist in A, B:
        # TODO(chrisburr) Explain
        # TODO(chrisburr) Add test for combinators of combinators
        if dist in Model._current_model._silently_replace.values():
            # We need to copy the items to a list as we're deleting items from
            # the dictionary
            for key, value in list(Model._current_model._silently_replace.items()):
                if value == dist:
                    del Model._current_model._silently_replace[key]
                    Model._current_model._silently_replace[key] = X
                    if dist in Model._current_model._description:
                        del Model._current_model._description[dist]

        else:
            Model._current_model._silently_replace[dist] = X
            del Model._current_model._description[dist]

    return X
