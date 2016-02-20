import numpy as np
import tensorprob as tp
import scipy.stats as st
from numpy.testing import assert_array_almost_equal, assert_array_equal


def make_normal():
    mu = tp.Scalar('mu')
    sigma = tp.Scalar('sigma', lower=0)
    X = tp.Normal(mu, sigma)
    return mu, sigma, X


def test_init():
    with tp.Model():
        mu, sigma, X = make_normal()
        assert(X.mu is mu)
        assert(X.sigma is sigma)


def test_pdf():
    with tp.Model() as m:
        mu, sigma, X = make_normal()
        m.observed(X)

    xs = np.linspace(-5, 5, 100)
    out1 = st.norm.pdf(xs, 0, 1)
    m.assign({
        mu: 0,
        sigma: 1
    })
    out2 = m.pdf(xs)
    assert_array_almost_equal(out1, out2, 16)


def test_logp():
    with tp.Model() as m:
        mu, sigma, X = make_normal()
        m.observed(X)

    xs = np.linspace(-5, 5, 100)
    out1 = st.norm.logpdf(xs, 0, 1)
    m.assign({
        mu: 0,
        sigma: 1
    })
    out2 = m.logp(xs)
    assert_array_equal(out1, out2)


def test_cdf():
    with tp.Model() as model:
        mu, sigma, X = make_normal()
        model.observed(X)

    xs = np.linspace(-5, 5, 100)
    out1 = st.norm.cdf(xs, 0, 1)

    # There is currently no proper way to access the cdf so manually run
    # tensorflow for now
    model.assign({
        mu: 0,
        sigma: 1,
    })
    feed_dict = model._prepare_model([[0]])
    out2 = model.session.run(X.cdf(xs), feed_dict=feed_dict)

    assert_array_almost_equal(out1, out2, 16)
