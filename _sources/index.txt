
TensorProb documentation
======================================

.. toctree::
   :hidden:

   self

You've found the documentation for **TensorProb**, a probabilistic programming
framework based on TensorFlow_.

.. _Tensorflow: https://www.tensorflow.org/

**TensorProb is currently under construction! Expect things to break!**

We are working on implementing the following features:

 - High flexibility in defining the statistical model
 - Models are defined in a self-contained `with` block
 - Seamless switching between frequentist and bayesian paradigms
 - Finding the maximum likelihood estimate or MAP estimate using a variety of optimizers
 - Flexible sampling using different MCMC backends
 - An extensive library of probability distributions
 - Analytic and numeric marginalization of probability distributions to support missing data and physical boundaries
 - Convolution of probability distributions
 - Functions for calculating confidence and credible intervals
 - Functions for hypothesis testing

Benefits of using TensorFlow as a backend include

 - Fast evaluation of the model using multiple CPU threads and/or GPUs
 - Defining new probability distributions using symbolic variables in
   Python
 - Possibility to write new optimized operators in C++ and load them
   dynamically

The API documentation
---------------------

.. toctree::
   :maxdepth: 2

   api
