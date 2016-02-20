
import numpy as np
from tensorprob import Model, Scalar, Normal

with Model() as model:
    mu = Scalar()
    sigma = Scalar(lower=0)
    X = Normal(mu, sigma)
    model.observed(X)

model.assign({
    mu: 1,
    sigma: 2,
})

# Create dataset
data = np.random.normal(0, 1, 1000)
model.fit(data)

import matplotlib.pyplot as plt
xs = np.linspace(-5, 5, 200)
plt.hist(data, bins=20, histtype='step', color='k', normed=True)
plt.plot(xs, model.pdf(xs), 'b-')
plt.show()
