import numpy as np
from nose.tools import raises

import tensorprob as tp


@raises(tp.model.ModelError)
def test_scalar_creation_outside_with():
    tp.Scalar('mu')


@raises(tp.model.ModelError)
def test_distribution_creation_outside_with():
    with tp.Model():
        mu = tp.Scalar('mu')
        sigma = tp.Scalar('sigma', lower=0)
    tp.Normal(mu, sigma)


@raises(tp.model.ModelError)
def test_nesting_models():
    with tp.Model():
        with tp.Model():
            pass


def test_track_variables():
    with tp.Model() as model:
        mu = tp.Scalar('mu')
    assert(mu in model.components)


def test_untrack_variables():
    with tp.Model() as model:
        mu = tp.Scalar('mu')
    model.untrack_variable(mu)
    assert(mu not in model.components)


def test_creation():
    model = tp.Model()
    with model:
        pass


def test_fit():
    model = tp.Model()
    with model:
        mu = tp.Scalar()
        sigma = tp.Scalar()
        X = tp.Normal(mu, sigma)

    model.observed(X)
    model.assign({mu: 2, sigma: 2})
    np.random.seed(0)
    data = np.random.normal(0, 1, 100)
    results = model.fit(data)
    assert results.success
