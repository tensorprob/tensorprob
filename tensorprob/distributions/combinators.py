import tensorflow as tf

from .. import config
from ..distribution import Distribution
from ..model import Model, Region
from ..utilities import is_finite


@Distribution
def Mix2(f, A, B, name=None):
    # TODO(chrisburr) Check if f is bounded between 0 and 1?
    X = tf.placeholder(config.dtype, name=name)

    a_logp, a_integral, a_bounds = Model._current_model._description[A]
    b_logp, b_integral, b_bounds = Model._current_model._description[B]

    Distribution.logp = tf.log(f*tf.exp(a_logp) + (1-f)*tf.exp(b_logp))

    def integral(lower, upper):
        a_integrals = []
        # if is_finite(bounds[0].lower) and is_finite(bounds[0].upper)
        for a_lower, a_upper in a_bounds:
            # Ignore this region if it's outside the current limits
            if a_upper < lower or upper < a_lower:
                continue
            # Else keep the region, tightening the edges as reqired
            a_integrals.append(a_integral(max(a_lower, lower), min(a_upper, upper)))

        a_integrals = tf.add_n(a_integrals) if a_integrals else tf.constant(1, config.dtype)

        b_integrals = []
        for b_lower, b_upper in b_bounds:
            # Ignore this region if it's outside the current limits
            if b_upper < lower or upper < b_lower:
                continue
            # Else keep the region, tightening the edges as reqired
            b_integrals.append(b_integral(max(b_lower, lower), min(b_upper, upper)))

        b_integrals = tf.add_n(b_integrals) if b_integrals else tf.constant(1, config.dtype)

        return f*a_integrals + (1-f)*b_integrals

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
