
import numpy as np
from tensorprob import Model, Normal, Uniform

def test_mcmc():
    with Model() as model:
        x = Normal(0, 1)

    np.random.seed(0)
    model.observed()
    model.initialize({
        x: 0.0,
    })
    out = model.mcmc(samples=100)

