from __future__ import absolute_import

import numpy as np
from .base import BaseOptimizer
from ..optimization_result import OptimizationResult

class Min_Func:
    def __init__(self, f, names):
        from iminuit.util import make_func_code
        self.f = f
        self.func_code = make_func_code(names)
        self.func_defaults = None

    def __call__(self, *args):
        return self.f(args)


class MigradOptimizer(BaseOptimizer):

    def __init__(self, verbose=False, **kwargs):
        try:
            import iminuit
        except ImportError:
            raise ImportError("The iminuit package needs to be installed in order to use MigradOptimizer")
        self.iminuit = iminuit

        self.verbose=verbose

        super(MigradOptimizer, self).__init__(**kwargs)

    def minimize_impl(self, objective, gradient, inits, bounds):

        x0 = dict()

        print_level = 2 if self.verbose else 0

        names = ['var_{}'.format(i) for i in range(len(inits))]

        all_kwargs = dict()

        for n, x in zip(names, inits):
            all_kwargs[n] = x
            # TODO use a method to set this correctly
            all_kwargs['error_' + n] = 1

        if bounds:
            for n, b in zip(names, bounds):
                all_kwargs['limit_' + n] = b

        if gradient:
            def mygrad_func(*x):
                out = gradient(x)
                return out
        else:
            mygrad_func = None

        def objective_(x):
            val = objective(x)
            if np.isnan(val) or np.isinf(val):
                return 1e10
            return val

        m = self.iminuit.Minuit(
            Min_Func(objective_, names),
            grad_fcn=mygrad_func,
            print_level=print_level,
            errordef=1,
            **all_kwargs
        )

        m.set_strategy(2)
        a, b = m.migrad()

        x = []
        for name in names:
            x.append(m.values[name])

        return OptimizationResult(
            x=x,
            err=m.errors,
            func=a['fval'],
            edm=a['edm'],
            calls=a['nfcn'],
            success=a['is_valid'],
            has_valid_parameters=a['has_valid_parameters'],
        )

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, session):
        self._session = session
