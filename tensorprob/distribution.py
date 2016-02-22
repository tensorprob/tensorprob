import tensorflow as tf
from collections import namedtuple

import numpy as np

from . import model
from . import utilities


Region = namedtuple('Region', ['lower', 'upper'])


class DistributionError(Exception):
    pass


def _parse_bounds(lower, upper, bounds):
    if not bounds:
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
        return [Region(lower, upper)]

    # Convert bounds to be a list of Region tuples
    if not isinstance(bounds[0], tuple):
        bounds = utilities.grouper(bounds)
    return [Region(*b) for b in bounds]


def Distribution(distribution_init):
    def f(*args, lower=None, upper=None, bounds=None, name=None):
        if bounds and (lower is not None or upper is not None):
            # Only allow the use of lower/upper if bounds is None
            raise DistributionError(
                "'lower'/'upper' can't be used in combination with 'bounds'"
            )

        name = name or utilities.generate_name(distribution_init)

        Distribution.logp = None
        Distribution.integral = None
        variables = distribution_init(*args, name=name)

        if lower is not None and upper is not None:
            Distribution.logp = Distribution.logp - tf.log(Distribution.integral(lower, upper))

        if Distribution.logp is None:
            raise DistributionError('Distributions must define logp')

        if Distribution.integral is None:
            raise NotImplementedError('Numeric integrals are not yet supported')

        if isinstance(variables, tuple):
            if isinstance(bounds[0][0], Region):
                if len(variables) != len(bounds):
                    raise DistributionError(
                        "Either a single set of 'bounds' must be provided or "
                        "the number of bounds ({0}) must equal the "
                        "dimensionality of the distribution ({1})"
                        .format(len(bounds), len(variables))
                    )
            else:
                # Set the same bounds for all variables
                bounds = [bounds]*len(variables)
        else:
            # We have a 1D distribution so convert variables/bounds to tuples
            variables = (variables,)
            bounds = (bounds,)

        for variable, b in zip(variables, bounds):
            model.Model.current_model._description[variable] = model.Description(
                Distribution.logp, Distribution.integral, _parse_bounds(lower, upper, b)
            )

        return variable
    return f
