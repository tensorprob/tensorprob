from collections import Iterable

import numpy as np
import tensorflow as tf

from . import config
from . import utilities
from .model import Description, Model, ModelError, Region


class DistributionError(Exception):
    pass


def _parse_bounds(lower, upper, bounds):
    if not bounds:
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
        return [Region(lower, upper)]

    bounds = [Region(*b) for b in bounds]
    if None in utilities.flatten(bounds):
        raise ValueError
    return bounds


def Distribution(distribution_init):
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
        variables = distribution_init(*args, name=name)

        # One dimensional distributions return a value, convert it to a tuple
        if not isinstance(variables, tuple):
            variables = (variables,)

        # Ensure the distribution has set the required properties
        if Distribution.logp is None:
            raise DistributionError('Distributions must define logp')

        if Distribution.integral is None:
            raise NotImplementedError('Numeric integrals are not yet supported')

        # Parse the bounds to be a list of lists of Regions
        try:
            if len(variables) == len(bounds) and isinstance(bounds[0][0], Iterable):
                bounds = [_parse_bounds(lower, upper, b) for b in bounds]
            else:
                # Set the same bounds for all variables
                bounds = [_parse_bounds(lower, upper, bounds)]*len(variables)
        except Exception:
            raise ValueError("Failed to parse 'bounds'")

        # Force the logp to zero where the distribution is invalid
        for var, bound in zip(variables, bounds):
            # numpy can't check Tensor objects so check the class type first
            if not isinstance(bound[0].lower, tf.Tensor) and np.isneginf(bound[0].lower) \
                    and not isinstance(bound[0].upper, tf.Tensor) and np.isposinf(bound[0].upper):
                continue

            # Force logp to negative infinity when outside the allowed bounds
            conditions = [tf.logical_and(tf.greater(var, l), tf.less(var, u)) for l, u in bound]

            is_inside_bounds = conditions[0]
            for condition in conditions[1:]:
                is_inside_bounds = tf.logical_or(is_inside_bounds, condition)

            Distribution.logp = tf.select(
                is_inside_bounds,
                Distribution.logp,
                tf.fill(tf.shape(var), config.dtype(-np.inf))
            )

        # Add the new variables to the model description
        for variable, bound in zip(variables, bounds):
            Model.current_model._description[variable] = Description(
                Distribution.logp, Distribution.integral, bound
            )

        return variable if len(variables) == 1 else variables
    return f
