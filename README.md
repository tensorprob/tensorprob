# TensorProb

[![Build Status](https://travis-ci.org/ibab/tensorprob.svg?branch=master)](https://travis-ci.org/ibab/tensorprob) [![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://ibab.github.io/tensorprob)


A probabalistic graphical modeling framework based on [TensorFlow](https://github.com/tensorflow/tensorflow).

TensorProb is a library that allows you to construct complex multi-dimensional probability distributions
from basic primitives and to perform inference on data.

The posterior distribution or likelihood are constructed and evaluated using TensorFlow,
which means you can make use of multiple CPU cores and GPUs simultaneously.
This also makes it very easy to add new custom probability distributions, or to
debug your model if it's not doing what you expect.

