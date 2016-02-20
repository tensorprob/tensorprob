import numpy as np
from nose.tools import raises
import scipy.stats as st

import tensorprob as tp


def test_creation():
    model = tp.Model()
    with model:
        pass


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


def test_observed():
    with tp.Model() as model:
        mu = tp.Scalar()
        sigma = tp.Scalar()
        X = tp.Normal(mu, sigma)
    model.observed(X)
    assert(X in model._observed)


@raises(ValueError)
def test_observed_erorr_on_non_distribution():
    with tp.Model() as model:
        mu = tp.Scalar()
        sigma = tp.Scalar()
        X = tp.Normal(mu, sigma)
    model.observed(X, int(42))


@raises(tp.model.ModelError)
def test_prepare_without_observed():
    with tp.Model() as model:
        pass
    model._prepare_model([])


@raises(tp.model.ModelError)
def test_prepare_with_incorrect_parameters():
    with tp.Model() as model:
        mu = tp.Scalar()
        sigma = tp.Scalar()
        X = tp.Normal(mu, sigma)
    model.observed(X)
    model._prepare_model([])


def test_nll():
    with tp.Model() as model:
        mu = tp.Scalar()
        sigma = tp.Scalar()
        X = tp.Normal(mu, sigma)
    model.observed(X)

    xs = np.linspace(-5, 5, 100)
    out1 = -sum(st.norm.logpdf(xs, 0, 1))
    model.assign({
        mu: 0,
        sigma: 1
    })
    out2 = model.nll(xs)
    assert(out1, out2)


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
