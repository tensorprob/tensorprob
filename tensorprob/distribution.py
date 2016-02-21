from collections import namedtuple

import numpy as np

from .. import model
from .. import utilities


Region = namedtuple('Region', ['lower', 'upper'])


class DistributionError(Exception):
    pass


def Distribution(distribution_init):
    def f(*args, lower=None, upper=None, bounds=None, name=None):
        # TODO(chrisburr) Add a way of specifying different bounds for multiple
        # variables
        # Parse the bounds
        if bounds is None:
            lower = -np.inf if lower is None else lower
            upper = np.inf if upper is None else upper
            bounds = [Region(lower, upper)]
        elif lower is not None or upper is not None:
            # Only allow the use of lower/upper if bounds is None
            raise DistributionError(
                "'lower'/'upper' can't be used in combination with 'bounds'"
            )
        else:
            # TODO(chrisburr) This should check that the bounds are a list of
            # Regions else it needs to be parsed
            raise NotImplementedError

        name = name or utilities.generate_name(distribution_init)

        Distribution.logp = None
        Distribution.integral = None
        variables = distribution_init(*args, name=name)

        if Distribution.logp is None:
            raise DistributionError('Distributions must define logp')

        if Distribution.integral is None:
            raise NotImplementedError('Numeric integrals are not yet supported')

        if not isinstance(variables, tuple):
            variables = tuple(variables)

        for variable in variables:
            model.Model._description[variable] = model.Description(
                Distribution.logp, Distribution.integral, bounds
            )

        return variable
    return f
