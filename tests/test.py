def test_Model_creation():
    import tensorprob as tp
    model = tp.Model()
    with model:
        pass

def test_Model_fit():
    import tensorprob as tp
    import numpy as np
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
