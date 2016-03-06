import tensorflow as tf

from .. import config
from ..distribution import Distribution, DistributionError
from ..model import Model
from ..utilities import Description, find_common_bounds, Region


def _integrate_component(sub_bounds, sub_integral):
    if sub_bounds:
        return tf.add_n([sub_integral(l, u) for l, u in sub_bounds])
    else:
        return tf.constant(1, config.dtype)


def _recurse_deps(X, f_scale, bounds):
    """Modify the fractional contribution and bounds of `X` in the model

    Recursivly iterates through the model, scaling `X` and all it's
    dependencies by `f_scale`.

    # Arguments:
        X: tensorflow.Tensor
            The variable to rewrite to modify
        f_scale: tensorflow.Tensor
            The fraction to scale `X`'s pdf by
        bounds: list of `Region` objects
            The boundaries to add to `X`
    """
    logp, integral, x_bounds, frac, deps = Model._current_model._full_description[X]

    x_bounds = find_common_bounds(x_bounds, bounds)
    frac = frac*f_scale

    Model._current_model._full_description[X] = Description(logp, integral, x_bounds, frac, deps)

    for dep in deps:
        _recurse_deps(dep, f_scale, x_bounds)


@Distribution
def Mix2(f, A, B, name=None):
    return _MixN([f, 1-f], [A, B], name)


@Distribution
def Mix3(f1, f2, A, B, C, name=None):
    return _MixN([f1*f2, (1-f1)*f2, (1-f2)], [A, B, C], name)


@Distribution
def MixN(fs, Xs, name=None):
    # Eqivilent to calling Mix2(Mix2(Mix2(Mix2(A, B), C), D), E)
    assert len(fs)+1 == len(Xs)
    fractions = [fs[0], 1-fs[0]]
    for f in fs[1:]:
        fractions = [f*frac for frac in fractions]
        fractions.append(1-f)
    return _MixN(fractions, Xs, name)


def _MixN(fractions, Xs, name=None):
    # Convert Xs to be iterable incase we are dealing with the 1D case
    Xs = [(x,) if isinstance(x, tf.Tensor) else tuple(x) for x in Xs]

    # Ensure all subdistributions have the same dimensionality
    if len(set(len(x) for x in Xs)) != 1:
        raise DistributionError("All components passed to 'MixN' must have "
                                "the same dimensionality.")

    n_dims = len(Xs[0])

    nd_X = tuple(tf.placeholder(config.dtype, name=name) for i in range(n_dims))
    nd_mix_bounds = Distribution.bounds(n_dims)

    current_model = Model._current_model

    full_pdf = []
    all_integrals = []
    for dists, f_scale in zip(Xs, fractions):
        nd_logps = []
        nd_bounds = []
        nd_integrals = []
        nd_normalisation_1 = []
        for dist, mix_bounds, X in zip(dists, nd_mix_bounds, nd_X):
            logp, integral, bounds, frac, _ = current_model._description[dist]
            bounds = find_common_bounds(mix_bounds, bounds)
            normalisation_1 = _integrate_component(bounds, integral)

            nd_logps.append(logp-tf.log(normalisation_1))
            nd_bounds.append(bounds)
            nd_integrals.append(integral)
            nd_normalisation_1.append(normalisation_1)

            # Modify the current model to recognize that 'deps' has been removed
            if dist in current_model._silently_replace.values():
                # We need to copy the items to a list as we're adding items
                for key, value in list(current_model._silently_replace.items()):
                    if value != dist:
                        continue
                    current_model._silently_replace[value] = X
                    current_model._silently_replace[key] = X
                    if dist in current_model._description:
                        del current_model._description[dist]

            else:
                current_model._silently_replace[dist] = X
                del current_model._description[dist]

            _recurse_deps(dist, f_scale, bounds)

        full_pdf.append(f_scale*tf.exp(tf.add_n(nd_logps)))

        all_integrals.append((f_scale, nd_bounds, nd_integrals, nd_normalisation_1))

    # Set properties on Distribution
    Distribution.logp = tf.log(tf.add_n(full_pdf))

    def _integral(lower, upper):
        result = []
        for f_scale, nd_bounds, nd_integral, normalisation_1 in all_integrals:
            nd_normalisation_2 = []
            for bounds, integral in zip(nd_bounds, nd_integral):
                integral_bounds = find_common_bounds([Region(lower, upper)], bounds)
                nd_normalisation_2.append(_integrate_component(integral_bounds, integral))
            result.append(f_scale/tf.add_n(normalisation_1)*tf.add_n(nd_normalisation_2))
        return tf.add_n(result)

    Distribution.integral = _integral

    if n_dims == 1:
        Distribution.depends = [x[0] for x in Xs]
    else:
        Distribution.depends = Xs

    return nd_X
