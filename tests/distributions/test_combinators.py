from __future__ import division

import numpy as np

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

    # Include the data blinded by the Mix2 bounds as we use the len(norm1_data)
    norm1_data = np.random.normal(19, 2, 100000)
    norm1_data = norm1_data[
        ((6 < norm1_data) & (norm1_data < 17)) |
        ((18 < norm1_data) & (norm1_data < 21)) |
        ((22 < norm1_data) & (norm1_data < 36))
    ]

    data = np.concatenate([exp_data, norm1_data])
    data = data[((6 < data) & (data < 17)) | ((18 < data) & (data < 36))]

    model.fit(data)

    assert model.state[mu] - 19 < 5e-3
    assert model.state[sigma] - 2 < 5e-3
    assert model.state[a] - 0.1 < 5e-4
    assert model.state[f] - (len(norm1_data)/len(data)) < 5e-5
