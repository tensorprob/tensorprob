import scipy.stats as st
import numpy as np
import tensorprob as tp
from numpy.testing import assert_array_almost_equal


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
    m.assign({mu: 0, sigma: 1})
    out2 = m.pdf(xs)
    assert_array_almost_equal(out1, out2, 16)
