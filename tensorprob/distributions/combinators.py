from itertools import product

import tensorflow as tf

from .. import config
from ..distribution import Distribution
from ..model import Model, Region


@Distribution
def Mix2(f, A, B, name=None):
    # TODO(chrisburr) Check if f is bounded between 0 and 1?
    X = tf.placeholder(config.dtype, name=name)

    a_logp, a_integral, a_bounds = Model._current_model._description[A]
    b_logp, b_integral, b_bounds = Model._current_model._description[B]

    def _integral(mix_bounds, sub_bounds, sub_integral):
        # Calculate the normalised logp
        integrals = []
        for (mix_lower, mix_upper), (sub_lower, sub_upper) in product(mix_bounds, sub_bounds):
            # Ignore this region if it's outside the current limits
            if sub_upper <= mix_lower or mix_upper <= sub_lower:
                continue
            # Else keep the region, tightening the edges as reqired
            integrals.append(sub_integral(max(sub_lower, mix_lower), min(sub_upper, mix_upper)))
        return tf.add_n(integrals) if integrals else tf.constant(1, config.dtype)

    bounds = Distribution.bounds(1)[0]
    a_normalisation_1 = _integral(bounds, a_bounds, a_integral)
    b_normalisation_1 = _integral(bounds, b_bounds, b_integral)

    Distribution.logp = tf.log(
        f*tf.exp(a_logp)/a_normalisation_1 +
        (1-f)*tf.exp(b_logp)/b_normalisation_1
    )

    def integral(lower, upper):
        a_normalisation_2 = _integral([Region(lower, upper)], a_bounds, a_integral)
        b_normalisation_2 = _integral([Region(lower, upper)], b_bounds, b_integral)

        return f/a_normalisation_1*a_normalisation_2 + (1-f)/b_normalisation_1*b_normalisation_2

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
                    Model._current_model._silently_replace[value] = X
                    Model._current_model._silently_replace[key] = X
                    if dist in Model._current_model._description:
                        del Model._current_model._description[dist]

        else:
            Model._current_model._silently_replace[dist] = X
            del Model._current_model._description[dist]

    return X
