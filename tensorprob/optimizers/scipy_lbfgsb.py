
import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b

from ..optimization_result import OptimizationResult
from .base import BaseOptimizer


class ScipyLBFGSBOptimizer(BaseOptimizer):

    def __init__(self, verbose=False, callback=None, m=10, factr=1e3, pgtol=1e-3, **kwargs):
        self.verbose = verbose
        self.callback = callback
        self.m = m
        self.factr = factr
        self.pgtol = pgtol
        super(ScipyLBFGSBOptimizer, self).__init__(**kwargs)

    def minimize_impl(self, objective, gradient, inits, bounds):
        if gradient is None:
            approx_grad = True
        else:
            approx_grad = False

        self.niter = 0
        def callback(xs):
            self.niter += 1
            if self.verbose:
                if self.niter % 50 == 0:
                    print('iter  ', '\t'.join([x.name.split(':')[0] for x in variables]))
                print('{: 4d}   {}'.format(self.niter, '\t'.join(map(str, xs))))
            if self.callback is not None:
                self.callback(xs)

        results = fmin_l_bfgs_b(
                objective,
                inits,
                m=self.m,
                fprime=gradient,
                factr=self.factr,
                pgtol=self.pgtol,
                callback=callback,
                approx_grad=approx_grad,
                bounds=bounds,
        )

        ret = OptimizationResult()
        ret.x = results[0]
        ret.func = results[1]
        ret.niter = results[2]['nit']
        ret.calls = results[2]['funcalls']
        ret.message = results[2]['task'].decode().lower()
        ret.success = results[2]['warnflag'] == 0

        return ret
