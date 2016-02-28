import numpy as np
import tensorflow as tf
from nose.tools import raises
from numpy.testing import assert_almost_equal
import scipy.stats as st

import tensorprob as tp


def test_creation():
    model = tp.Model()
    with model:
        pass


@raises(tp.model.ModelError)
def test_scalar_creation_outside_with():
    tp.Parameter(name='mu')


@raises(tp.model.ModelError)
def test_observed_in_model_block():
    with tp.Model():
        mu = tp.Parameter()
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
        tp.observed(X)


@raises(tp.model.ModelError)
def test_initalize_in_model_block():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter(lower=0)
        tp.Normal(mu, sigma)
        model.initialize({
            mu: 5,
            sigma: 2
        })


@raises(tp.model.ModelError)
def test_observed_in_model_block():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
    model.observed(X)
    model.initialize({
        mu: 5,
        sigma: 2
    })
    model.initialize({
        mu: 5,
        sigma: 2
    })


@raises(tp.model.ModelError)
def test_initialise_before_observed():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter(lower=0)
        tp.Normal(mu, sigma)
    model.initialize({
        mu: 5,
        sigma: 2
    })


@raises(tp.model.ModelError)
def test_initialise_with_too_few_variables():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
    model.observed(X)
    model.initialize({
        mu: 5
    })


@raises(ValueError)
def test_initialise_with_list():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
    model.observed(X)
    model.initialize([5, 2])


@raises(ValueError)
def test_initialise_with_invalid_keys():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
    model.observed(X)
    model.initialize({
        'Parameter_1': 5,
        'Parameter_2': 2
    })


def test_distribution_creation_global_graph():
    # Distribution creation doesn't modify the global graph
    before = tf.get_default_graph().as_graph_def()
    with tp.Model():
        tp.Parameter()
    after = tf.get_default_graph().as_graph_def()
    assert before == after


def test_internal_graph_no_growth():
    # Calling assign or fit doesn't grow the execution graph
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)

    model.observed(X)
    model.initialize({
        mu: 1,
        sigma: 1,
    })

    before = model.session.graph_def
    model.assign({
        mu: 0,
        sigma: 2,
    })
    model.fit([1, 2, 3])
    after = model.session.graph_def

    assert before == after


@raises(tp.model.ModelError)
def test_distribution_creation_outside_with():
    with tp.Model():
        mu = tp.Parameter(name='mu')
        sigma = tp.Parameter(name='sigma', lower=0)
    tp.Normal(mu, sigma)


@raises(tp.model.ModelError)
def test_nesting_models():
    with tp.Model():
        with tp.Model():
            # Make coveralls ignore this line
            raise NotImplementedError


def test_observed():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter()
        X = tp.Normal(mu, sigma)
    model.observed(X)
    assert(X in model._observed)


@raises(ValueError)
def test_observed_erorr_on_non_distribution():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter()
        X = tp.Normal(mu, sigma)
    model.observed(X, int(42))


@raises(tp.model.ModelError)
def test_prepare_without_observed():
    with tp.Model() as model:
        mu = tp.Parameter()
    model.initialize({mu: 42})


def test_nll():
    with tp.Model() as model:
        mu = tp.Parameter()
        sigma = tp.Parameter()
        X = tp.Normal(mu, sigma)
    model.observed(X)

    xs = np.linspace(-5, 5, 100)
    model.initialize({
        mu: 0,
        sigma: 1
    })
    out1 = model.nll(xs)
    out2 = -sum(st.norm.logpdf(xs, 0, 1))
    assert_almost_equal(out1, out2, 10)


def test_fit():
    model = tp.Model()
    with model:
        mu = tp.Parameter(lower=-5, upper=5)
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)

    model.observed(X)
    model.initialize({mu: 2, sigma: 2})
    np.random.seed(0)
    data = np.random.normal(0, 1, 100)
    results = model.fit(data)
    assert results.success


def test_assign():
    model = tp.Model()
    with model:
        mu = tp.Parameter(lower=-5, upper=5)
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)

    model.observed(X)
    feed = {mu: 42, sigma: 1}
    model.initialize(feed)
    model.assign({mu: 1, sigma: 0})
    model.assign({mu: -100, sigma: 0})
    assert model.state == {mu: -100, sigma: 0}
    model.assign(feed)
    assert model.state == feed


@raises(tp.model.ModelError)
def test_assign_in_model_block():
    model = tp.Model()
    with model:
        mu = tp.Parameter(lower=-5, upper=5)
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
        model.assign({mu: 1, sigma: 0})

    model.observed(X)
    model.initialize({mu: 42, sigma: 1})


@raises(tp.model.ModelError)
def test_assign_in_before_observed():
    model = tp.Model()
    with model:
        mu = tp.Parameter(lower=-5, upper=5)
        sigma = tp.Parameter(lower=0)
        tp.Normal(mu, sigma)

    model.assign({mu: 1, sigma: 0})


@raises(tp.model.ModelError)
def test_assign_in_before_initalize():
    model = tp.Model()
    with model:
        mu = tp.Parameter(lower=-5, upper=5)
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
        model.assign({mu: 1, sigma: 0})

    model.observed(X)
    model.assign({mu: 1, sigma: 0})


@raises(ValueError)
def test_assign_empty_dict():
    model = tp.Model()
    with model:
        mu = tp.Parameter(lower=-5, upper=5)
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
    model.observed(X)
    model.assign({})


@raises(ValueError)
def test_assign_wrong_container():
    model = tp.Model()
    with model:
        mu = tp.Parameter(lower=-5, upper=5)
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
    model.observed(X)
    model.assign([1, 2, 3])


@raises(tp.model.ModelError)
def test_observed_in_model():
    model = tp.Model()
    with model:
        mu = tp.Parameter(lower=-5, upper=5)
        sigma = tp.Parameter(lower=0)
        X = tp.Normal(mu, sigma)
        model.observed(X)


def test_variable_independent():
    with tp.Model() as model:
        a = tp.Parameter()
        b = tp.Parameter()
    model.observed(b)
    model.initialize({
        a: 1,
    })
    model.fit([1, 2, 3])
