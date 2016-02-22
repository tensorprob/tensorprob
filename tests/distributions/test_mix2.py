
import numpy as np
from tensorprob import Model, Uniform, Normal, Mix2

def test_mix2():
    with Model() as model:
        mu1 = Uniform()
        mu2 = Uniform()
        sigma1 = Uniform(lower=0)
        sigma2 = Uniform(lower=0)
        f = Uniform(lower=0, upper=1)

        X = Mix2(
            f,
            Normal(mu1, sigma1),
            Normal(mu2, sigma2),
        )
    model.observed(X)
    model.initialize({
        f: 0.5,
        mu1: -1,
        mu2: 1,
        sigma1: 1,
        sigma2: 2,
    })
    model.fit([1,2,3])
