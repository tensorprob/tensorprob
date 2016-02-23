
import numpy as np
from numdifftools import Hessian


def fisher(model, variables=None):
    '''
    Calculates the covariance matrix of the variables, given the model.
    The model needs to have been optimized for this to work.
    '''

    if variables is None:
        keys = model._hidden.keys()
        variables = model._hidden.values()

    def func(xs):
        feed = { k: v for k, v in zip(variables, xs) }
        out = model.session.run(model._nll, feed_dict=feed)
        if np.isnan(out):
            return -1e40
        return out

    x = model.session.run(list(model._hidden.values()))
    hess = Hessian(func)(x)
    cov = np.linalg.inv(hess)

    result = dict()

    for i, v1 in enumerate(keys):
        result[v1] = dict()
        for j, v2 in enumerate(keys):
            result[v1][v2] = cov[i,j]

    return result
