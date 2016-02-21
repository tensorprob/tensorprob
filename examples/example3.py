import numpy as np
from tensorprob import Model, Parameter, Normal

# Define the model
with Model() as model:
    mu = Parameter()
    sigma = Parameter(lower=0)
    X = Normal(mu, sigma)

# Declare variables for which we have data
model.observed(X)

# Set the initial values
model.assign({
    mu: 10,
    sigma: 10,
})

# Create dataset with Numpy
data = np.random.normal(0, 1, 1000)

# Perform the fit
print(model.fit(data))

import matplotlib.pyplot as plt
xs = np.linspace(-5, 5, 200)
plt.hist(data, bins=20, histtype='step', color='k', normed=True)
plt.plot(xs, model.pdf(xs), 'b-')
plt.show()
