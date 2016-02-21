import numpy as np
from tensorprob import Model, Parameter, Normal

with Model() as model:
    mu = Parameter()
    sigma = Parameter(lower=0)

    X = Normal(mu, sigma)

model.observed(X)

model.assign({
    mu: 0,
    sigma: 1,
})

import matplotlib.pyplot as plt
xs = np.linspace(-5, 5, 200)
plt.plot(xs, model.pdf(xs), 'b-')
#plt.show()
plt.clf()

data = np.random.normal(0, 1, 1000)

from mpl_toolkits.mplot3d import Axes3D
ax = plt.gcf().add_subplot(111, projection='3d')

mu_, sigma_ = np.meshgrid(np.linspace(-.5, .5, 20), np.linspace(.5, 1.5, 20))

i = 0

@np.vectorize
def nll(mu_, sigma_):
    global i
    print(i)
    i += 1
    model.assign({mu: mu_, sigma: sigma_})
    return model.nll(data)

ax.plot_wireframe(mu_, sigma_, nll(mu_, sigma_))
plt.show()
