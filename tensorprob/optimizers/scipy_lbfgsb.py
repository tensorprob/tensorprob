
import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b

from .. import config
from ..optimization_result import OptimizationResult
from .base import BaseOptimizer


class ScipyLBFGSBOptimizer(BaseOptimizer):

    def __init__(self, session=None, verbose=False, callback=None):
        self._session = session or tf.Session()
        self.verbose = verbose
        self.callback = callback

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
            approx_grad = True

        if bounds:
            min_bounds = []
            for current in bounds:
                lower, upper = current

                # Variable limits should be ignored
                if isinstance(lower, tf.Tensor) or isinstance(upper, tf.Tensor):
                    raise NotImplementedError("ScipyLBFGSBOptimizer currently doesn't support variable bounds")

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

        self.niter = 0

        def callback(xs):
            self.niter += 1
            if self.verbose:
                if self.niter % 50 == 0:
                    print('iter  ', '\t'.join([x.name.split(':')[0] for x in variables]))
                print('{: 4d}   {}'.format(self.niter, '\t'.join(map(str, xs))))
            if self.callback is not None:
                self.callback(xs)

        results = fmin_l_bfgs_b(objective, inits, fprime=gradient_, callback=callback, approx_grad=approx_grad, bounds=min_bounds)

        ret = OptimizationResult()
        ret.x = results[0]
        ret.func = results[1]
        ret.niter = results[2]['nit']
        ret.calls = results[2]['funcalls']
        ret.message = results[2]['task'].decode().lower()
        ret.success = results[2]['warnflag'] == 0

        return ret
