import numpy as np
import tensorprob as tp
import scipy.stats as st
from numpy.testing import assert_array_almost_equal, assert_array_equal


def test_init():
    with tp.Model():
        X1 = tp.Uniform(lower=-1, upper=1)
        X2 = tp.Uniform(lower=-1)
        X3 = tp.Uniform(upper=1)
        X4 = tp.Uniform()
        X5 = tp.Uniform(lower=X1)
        X6 = tp.Uniform(upper=X2)
        X7 = tp.Uniform(lower=X1, upper=X2)


def test_pdf():
    with tp.Model() as m:
        X = tp.Uniform(-2, 2)

    m.observed(X)

    xs = np.linspace(-1, 1, 100)
    out1 = st.uniform.pdf(xs, -2, 4)
    out2 = m.pdf(xs)
    assert_array_almost_equal(out1, out2, 16)
