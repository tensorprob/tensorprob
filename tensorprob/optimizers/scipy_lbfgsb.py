
import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b

from .base import BaseOptimizer


class ScipyLBFGSBOptimizer(BaseOptimizer):

    def __init__(self, session=None, verbose=False):
        self.session = session or tf.Session()
        self.verbose = verbose

    def minimize(self, variables, cost, gradient=None, bounds=None):
        for v in variables:
            if not isinstance(v, tf.Variable):
                raise ValueError("Parameter {} is not a tensorflow variable".format(v))

        inits = self.session.run(variables)

        def objective(xs):
            feed_dict = { k: v for k, v in zip(variables, xs) }
            return self.session.run(cost, feed_dict=feed_dict)

        if gradient is None:
            def gradient(xs):
                feed_dict = { k: v for k, v in zip(variables, xs) }
                return self.session.run(gradient, feed_dict=feed_dict)
            approx_grad = False
        else:
            approx_grad = True

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

        if self.verbose:
            self.fev = 0
            def callback(xs):
                self.fev += 1
                print('{: 4d}   {}'.format(self.fev, '\t'.join(map(str, xs))))
            print('iter  ', '\t'.join([ x.name.split(':')[0] for x in variables]))
        else:
            callback = None

        results = fmin_l_bfgs_b(objective, inits, fprime=gradient, callback=callback, approx_grad=approx_grad, bounds=min_bounds)
        #self.session.run([v.assign(x) for v, x in zip(variables, results[0])])
        return results

