from collections import Iterable

import numpy as np
import tensorflow as tf

from . import config
from . import utilities
from .model import Model, ModelError
from .utilities import Description, Region


class DistributionError(Exception):
    pass


def _parse_bounds(num_dimensions, lower, upper, bounds):
    def _parse_bounds_1D(lower, upper, bounds):
        if not bounds:
            lower = -np.inf if lower is None else lower
            upper = np.inf if upper is None else upper
            return [Region(lower, upper)]

        bounds = [Region(*b) for b in bounds]
        if None in utilities.flatten(bounds):
            raise ValueError
        return bounds

    try:
        if num_dimensions == len(bounds) and isinstance(bounds[0][0], Iterable):
            bounds = [_parse_bounds_1D(lower, upper, b) for b in bounds]
        else:
            # Set the same bounds for all variables
            bounds = [_parse_bounds_1D(lower, upper, bounds)]*num_dimensions
    except Exception:
        raise ValueError("Failed to parse 'bounds'")
    else:
        return bounds


def Distribution(distribution_init):
    """Decorator used for defining distributions.

    The distribution function should return `N` tensorflow.Tensor objects
    where `N` is the number of dimensions the distribution has. Additionally
    distributions should set:
        Distribution.logp: tensorflow.Tensor
            The log probability.
        Distribution.integral: function
            A function with arguments (lower, upper) that returns the integral
            of the function between the specifed bounds.
        Distribution.depends: (optional) list of tensorflow.Tensor
            The sub-distributions of this distribution

    Most distributions are bounded automatically, however some distributions,
    such as combinators, require access to the bounds of the distribution.
    These can be accessed using `Distribution.bounds(N)`.

    # Arguments:
        distribution_init: function

    # Returns:
        f: function
            The decorated version of `distribution_init`
    """
    def f(*args, **kwargs):
        # Why legacy Python, why...
        lower = kwargs.get('lower')
        upper = kwargs.get('upper')
        bounds = kwargs.get('bounds', [])
        name = kwargs.get('name')

        if Model is None or tf.get_default_graph() is not Model.current_model._model_graph:
            raise ModelError(
                "Can't define distributions outside of a model block")

        if bounds and (lower is not None or upper is not None):
            raise DistributionError(
                "'lower'/'upper' can't be used incombination with 'bounds'")

        name = name or utilities.generate_name(distribution_init)

        Distribution.logp = None
        Distribution.integral = None
        Distribution.depends = []
        Distribution.bounds = lambda ndim: _parse_bounds(ndim, lower, upper, bounds)
        variables = distribution_init(*args, name=name)

        # One dimensional distributions return a value, convert it to a tuple
        if not isinstance(variables, tuple):
            variables = (variables,)

        # Ensure the distribution has set the required properties
        if Distribution.logp is None:
            raise DistributionError('Distributions must define logp')
        logp = Distribution.logp

        if Distribution.integral is None:
            raise NotImplementedError('Numeric integrals are not yet supported')

        # Parse the bounds to be a list of lists of Regions
        bounds = Distribution.bounds(len(variables))

        # Force logp to negative infinity when outside the allowed bounds
        for var, bound in zip(variables, bounds):
            logp = utilities.set_logp_to_neg_inf(var, logp, bound)

        # Add the new variables to the model description
        for variable, bound in zip(variables, bounds):
            description = Description(
                logp,
                Distribution.integral,
                bound,
                tf.constant(1, config.dtype),
                Distribution.depends
            )
            Model.current_model._description[variable] = description
            Model.current_model._full_description[variable] = description

        return variable if len(variables) == 1 else variables
    return f
