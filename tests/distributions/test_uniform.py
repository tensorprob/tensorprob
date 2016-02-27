import numpy as np
import tensorprob as tp
import scipy.stats as st
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import raises


def test_init():
    with tp.Model():
        X1 = tp.Uniform(lower=-1, upper=1)
        X2 = tp.Uniform(lower=-1)
        X3 = tp.Uniform(upper=1)
        X4 = tp.Uniform()
        X7 = tp.Uniform(lower=X1, upper=X2)


# @raises(ValueError)
# def test_uniform_fail_lower():
#     with tp.Model():
#         X1 = tp.Uniform()
#         X2 = tp.Uniform(lower=X1)


# @raises(ValueError)
# def test_uniform_fail_upper():
#     with tp.Model() as model:
#         X1 = tp.Uniform()
#         X2 = tp.Uniform(upper=X1)


def test_pdf():
    with tp.Model() as m:
        dummy = tp.Parameter()
        X = tp.Uniform(lower=dummy, upper=1)

    m.observed(X)
    m.initialize({dummy: -2})

    xs = np.linspace(-1, 1, 1)
    out1 = st.uniform.pdf(xs, -2, 3)
    out2 = m.pdf(xs)
    assert_array_almost_equal(out1, out2, 16)
