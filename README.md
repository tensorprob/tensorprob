# TensorProb

[![Build Status](https://travis-ci.org/ibab/tensorprob.svg?branch=master)](https://travis-ci.org/ibab/tensorprob)
[![Coverage Status](https://coveralls.io/repos/github/ibab/tensorprob/badge.svg?branch=master)](https://coveralls.io/github/ibab/tensorprob?branch=master)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://ibab.github.io/tensorprob)

TensorProb is a probabalistic graphical modeling framework based on
[TensorFlow](https://github.com/tensorflow/tensorflow).

It's a Python library that allows you to construct complex multi-dimensional
probability distributions from basic building blocks and to infer their
parameters from data.

Fitting a normal distribution to data is as simple as
```python
import tensorprob as tp

with tp.Model() as model:
    mu, sigma = Scalar(), Scalar(lower=0)
    X = tp.Normal(mu, sigma)
    model.bind_params([('X', X)])
```

The posterior distribution (or likelihood function) are constructed and
evaluated using TensorFlow, which means you can make use of multiple CPU cores
and GPUs simultaneously. This also makes it easy to add new custom probability
distributions, or to debug your model if it's not doing what you expect.

## Contributing to TensorProb

We happily accept contributions to the project!
Please have a look at [`CONTRIBUTING.md`](CONTRIBUTING.md) for instructions and guidelines for contributing.

