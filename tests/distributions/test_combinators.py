from __future__ import division

import numpy as np
import scipy.stats as st
from numpy.testing import assert_array_almost_equal

from tensorprob import Model, Parameter, Normal, Exponential, Mix2


def test_mix2_fit():
    with Model() as model:
        mu = Parameter()
        sigma = Parameter(lower=1)
        a = Parameter(lower=0)
        f = Parameter(lower=0, upper=1)

        X1 = Normal(mu, sigma, bounds=[(-np.inf, 21), (22, np.inf)])
        X2 = Exponential(a, bounds=[(-np.inf, 8), (10, np.inf)])
        X12 = Mix2(f, X1, X2, bounds=[(6, 17), (18, 36)])

    model.observed(X12)
    model.initialize({
        mu: 23,
        sigma: 1.2,
        a: 0.2,
        f: 0.3,
    })

    # Generate some data to fit
    np.random.seed(42)

    exp_data = np.random.exponential(10, 200000)
    exp_data = exp_data[(exp_data < 8) | (10 < exp_data)]

    # Include the data blinded by the Mix2 bounds as we use the len(norm_data)
    norm_data = np.random.normal(19, 2, 100000)
    norm_data = norm_data[
        ((6 < norm_data) & (norm_data < 17)) |
        ((18 < norm_data) & (norm_data < 21)) |
        ((22 < norm_data) & (norm_data < 36))
    ]

    data = np.concatenate([exp_data, norm_data])
    data = data[((6 < data) & (data < 17)) | ((18 < data) & (data < 36))]

    result = model.fit(data)

    # Check the fit was successful
    assert result.success
    assert abs(model.state[mu] - 19) < 5e-3
    assert abs(model.state[sigma] - 2) < 5e-3
    assert abs(model.state[a] - 0.1) < 5e-4
    assert abs(model.state[f] - (len(norm_data)/len(data))) < 5e-4


def test_mix2_fit_with_mix2_input():
    with Model() as model:
        mu = Parameter()
        sigma = Parameter(lower=1, upper=4)
        a = Parameter(lower=0.06)
        b = Parameter(lower=0)
        f_1 = Parameter(lower=0, upper=1)
        f_2 = Parameter(lower=0, upper=1)

        X1 = Normal(mu, sigma, bounds=[(-np.inf, 21), (22, np.inf)])
        X2 = Exponential(a, bounds=[(-np.inf, 8), (10, 27), (31, np.inf)])
        X12 = Mix2(f_1, X1, X2, bounds=[(6, 17), (18, 36)])

        X3 = Exponential(b)
        X123 = Mix2(f_2, X12, X3, bounds=[(6, 17), (18, 36)])

    model.observed(X123)
    model.initialize({
        mu: 23,
        sigma: 1.2,
        a: 0.2,
        b: 0.04,
        f_1: 0.3,
        f_2: 0.4
    })

    # Generate some data to fit
    np.random.seed(42)

    exp_1_data = np.random.exponential(10, 200000)
    exp_1_data = exp_1_data[
        (6 < exp_1_data) &
        ((exp_1_data < 8) | (10 < exp_1_data)) &
        ((exp_1_data < 17) | (18 < exp_1_data)) &
        ((exp_1_data < 27) | (31 < exp_1_data)) &
        (exp_1_data < 36)
    ]

    exp_2_data = np.random.exponential(20, 200000)
    exp_2_data = exp_2_data[
        (6 < exp_2_data) &
        ((exp_2_data < 17) | (18 < exp_2_data)) &
        (exp_2_data < 36)
    ]

    # Include the data blinded by the Mix2 bounds as we use the len(norm_data)
    norm_data = np.random.normal(19, 2, 100000)
    norm_data = norm_data[
        ((6 < norm_data) & (norm_data < 17)) |
        ((18 < norm_data) & (norm_data < 21)) |
        ((22 < norm_data) & (norm_data < 36))
    ]

    data = np.concatenate([exp_1_data, exp_2_data, norm_data])
    data = data[((6 < data) & (data < 17)) | ((18 < data) & (data < 36))]

    result = model.fit(data)

    # Check the fit was successful
    assert result.success
    assert abs(model.state[mu] - 19) < 3e-2
    assert abs(model.state[sigma] - 2) < 1e-3
    assert abs(model.state[a] - 0.1) < 1e-3
    assert abs(model.state[b] - 0.05) < 3e-4
    assert abs(model.state[f_1] - (len(norm_data)/(len(exp_1_data)+len(norm_data)))) < 5e-3
    assert abs(model.state[f_2] - ((len(exp_1_data)+len(norm_data))/len(data))) < 5e-4

    # Check if we can access the individual components
    xs = np.linspace(0, 41, 1001)

    def allowed_point(x, bounds):
        @np.vectorize
        def allowed_point(x):
            for l, u in bounds:
                if l < x and x < u:
                    return 1
            return 0
        return allowed_point(x)

    # Normal
    bounds = [(6, 17), (18, 21), (22, 36)]
    out1 = st.norm.pdf(xs, model.state[mu], model.state[sigma]) * allowed_point(xs, bounds)
    integral = sum(
        st.norm.cdf(u, model.state[mu], model.state[sigma]) -
        st.norm.cdf(l, model.state[mu], model.state[sigma])
        for l, u in bounds
    )
    out1 *= model.state[f_1] * model.state[f_2] / integral

    out2 = model[X1].pdf(xs)

    assert_array_almost_equal(out1, out2, 11)

    # Exponential 1
    bounds = [(6, 8), (10, 17), (18, 27), (31, 36)]
    out1 = st.expon.pdf(xs, 0, 1/model.state[a]) * allowed_point(xs, bounds)
    integral = sum(
        st.expon.cdf(u, 0, 1/model.state[a]) -
        st.expon.cdf(l, 0, 1/model.state[a])
        for l, u in bounds
    )
    out1 *= (1-model.state[f_1]) * model.state[f_2] / integral

    out2 = model[X2].pdf(xs)

    assert_array_almost_equal(out1, out2, 11)

    # Exponential 2
    bounds = [(6, 17), (18, 36)]
    out1 = st.expon.pdf(xs, 0, 1/model.state[b]) * allowed_point(xs, bounds)
    integral = sum(
        st.expon.cdf(u, 0, 1/model.state[b]) -
        st.expon.cdf(l, 0, 1/model.state[b])
        for l, u in bounds
    )
    out1 *= (1-model.state[f_2]) / integral

    out2 = model[X3].pdf(xs)

    assert_array_almost_equal(out1, out2, 11)
