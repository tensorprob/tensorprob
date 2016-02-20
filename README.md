# TensorProb

[![Build Status](https://img.shields.io/travis/ibab/tensorprob/master.svg)](https://travis-ci.org/ibab/tensorprob)
[![Coverage Status](https://img.shields.io/coveralls/ibab/tensorprob/master.svg)](https://coveralls.io/github/ibab/tensorprob?branch=master)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://ibab.github.io/tensorprob)

TensorProb is a probabalistic graphical modeling framework based on
[TensorFlow](https://github.com/tensorflow/tensorflow).

It's a Python library that allows you to construct complex multi-dimensional
probability distributions from basic building blocks and to infer their
parameters from data.

This is an example for fitting a normal distribution to data:
```python
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
model.fit(data)
```
The fitted distribution can be visualized with `model.pdf`
```python
import matplotlib.pyplot as plt
xs = np.linspace(-5, 5, 200)
plt.hist(data, bins=20, histtype='step', color='k', normed=True)
plt.plot(xs, model.pdf(xs), 'b-')
```
<div align="center"><img src="examples/example3.png" width="600px"/></div>


The posterior distribution (or likelihood function) are constructed and
evaluated using TensorFlow, which means you can make use of multiple CPU cores
and GPUs simultaneously. This also makes it easy to add new custom probability
distributions, or to debug your model if it's not doing what you expect.

## Contributing to TensorProb

We happily accept contributions to the project!
Please have a look at [`CONTRIBUTING.md`](CONTRIBUTING.md) for instructions and guidelines for contributing.

