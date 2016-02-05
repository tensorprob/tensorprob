# tensorfit

A likelihood framework based on [TensorFlow](https://github.com/tensorflow/tensorflow).

tensorfit is a library allows you to construct probility distributions, perform
inference on data, and visualize the results. The probability function is
constructed and evaluated using TensorFlow, which means your model can
automatically be evaluated on multiple CPU cores and GPUs.
This also makes it very easy to add new custom probability distributions, or to
debug your model if it's not doing what you expect.

