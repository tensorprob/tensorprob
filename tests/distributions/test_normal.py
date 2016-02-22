import numpy as np
import tensorprob as tp
import scipy.stats as st
from numpy.testing import assert_array_almost_equal, assert_array_equal


def make_normal():
    mu = tp.Parameter(name='mu')
    sigma = tp.Parameter(name='sigma', lower=0)
    X = tp.Normal(mu, sigma)
    return mu, sigma, X


#def test_init():
#    with tp.Model():
#        mu, sigma, X = make_normal()
#        assert(X.mu is mu)
#        assert(X.sigma is sigma)


def test_pdf():
    with tp.Model() as m:
        mu, sigma, X = make_normal()

    m.observed(X)

    xs = np.linspace(-5, 5, 100)
    out1 = st.norm.pdf(xs, 0, 1)
    m.initialize({
        mu: 0,
        sigma: 1
    })
    out2 = m.pdf(xs)
    assert_array_almost_equal(out1, out2, 16)


#def test_cdf():
#    xs = np.linspace(-5, 5, 100)
#    out1 = st.norm.cdf(xs, 0, 1)
#
#    with tp.Model() as model:
#        mu, sigma, X = make_normal()
#
#    out2 = model.session.run(X.cdf(xs), feed_dict={mu: 0, sigma: 1})
#
#    assert_array_almost_equal(out1, out2, 16)
