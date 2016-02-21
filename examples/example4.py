from tensorprob import Model, Parameter, Normal

with Model() as model:
    a = Parameter()
    b = Parameter()
    sigma = Parameter(lower=0)
    X = Parameter()
    y = Normal(a * X + b, sigma)

model.observed(X, y)
model.assign({
    a: 2,
    b: 2,
    sigma: 10,
})

import numpy as np
xs = np.linspace(0, 1, 100)
ys = 1 * xs + 0 +  np.random.normal(0, .1, len(xs))

print(model.fit(xs, ys))
import matplotlib.pyplot as plt
plt.plot(xs, ys, 'ro')
x_ = np.linspace(0, 1, 200)
plt.plot(x_, model.state[a] * x_ + model.state[b], 'b-')
plt.show()
