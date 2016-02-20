# TensorProb

[![Build Status](https://img.shields.io/travis/ibab/tensorprob/master.svg)](https://travis-ci.org/ibab/tensorprob)
[![Coverage Status](https://img.shields.io/coveralls/ibab/tensorprob/master.svg)](https://coveralls.io/github/ibab/tensorprob?branch=master)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://ibab.github.io/tensorprob)

TensorProb is a probabalistic graphical modeling framework based on
[TensorFlow](https://github.com/tensorflow/tensorflow).

It's a Python library that allows you to construct complex multi-dimensional
probability distributions from basic building blocks and to infer their
parameters from data.

Fitting a normal distribution to data is as simple as
```python
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
```
The resulting fit can be visualized with
```
import matplotlib.pyplot as plt
xs = np.linspace(-5, 5, 200)
plt.hist(data, bins=20, histtype='stepfilled', color='k')
plt.plot(xs, model.pdf(xs), 'b-')
```
![Plot](./examples/example3.png)


The posterior distribution (or likelihood function) are constructed and
evaluated using TensorFlow, which means you can make use of multiple CPU cores
and GPUs simultaneously. This also makes it easy to add new custom probability
distributions, or to debug your model if it's not doing what you expect.

## Contributing to TensorProb

We happily accept contributions to the project!
Please have a look at [`CONTRIBUTING.md`](CONTRIBUTING.md) for instructions and guidelines for contributing.

