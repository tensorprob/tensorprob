import numpy as np
import tensorflow as tf
from .. import config

class BaseOptimizer(object):

    def __init__(self, session=None):
        self._session = session or tf.Session()

    def minimize_impl(self, objective, gradient, inits, bounds):
        raise NotImplementedError

    def minimize(self, variables, cost, gradient=None, bounds=None):
        # Check if variables is iterable
        try:
            iter(variables)
        except TypeError:
            raise ValueError("Variables parameter is not iterable")

        for v in variables:
            if not isinstance(v, tf.Variable):
                raise ValueError("Parameter {} is not a tensorflow variable".format(v))

        inits = self._session.run(variables)

        def objective(xs):
            feed_dict = {k: v for k, v in zip(variables, xs)}
            # Cast just in case the user-supplied function returns something else
            val = np.float64(self._session.run(cost, feed_dict=feed_dict))
            if config.debug:
                print('objective', val, xs)
            return val

        if gradient is not None:
            def gradient_(xs):
                feed_dict = {k: v for k, v in zip(variables, xs)}
                # Cast just in case the user-supplied function returns something else
                val = np.array(self._session.run(gradient, feed_dict=feed_dict))
                if config.debug:
                    print('gradient', val, xs)
                return val
            approx_grad = False
        else:
            gradient_ = None

        if bounds:
            min_bounds = []
            for current in bounds:
                lower, upper = current

                # Translate inf to None, which is what this minimizer understands
                if lower is not None and np.isinf(lower):
                    lower = None
                if upper is not None and np.isinf(upper):
                    upper = None
                # This optimizer likes to evaluate the function at the boundaries.
                # Slightly move the bounds so that the edges are not included.
                if lower is not None:
                    lower = lower + 1e-10
                if upper is not None:
                    upper = upper - 1e-10

                min_bounds.append((lower, upper))
        else:
            min_bounds = None

        return self.minimize_impl(objective, gradient_, inits, min_bounds)


    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, session):
        self._session = session
