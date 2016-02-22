import numpy as np
from nose.tools import raises

from tensorprob import Model, Region, config, Distribution, DistributionError
from tensorprob.distribution import _parse_bounds
import tensorflow as tf


def get_fake_distribution(logp=-42, integral=-42, dimension=1):
    if logp == -42:
        def logp():
            return 1

    if integral == -42:
        def integral():
            return 2

    def FakeDistribution(name=None):
        FakeDistribution.name = name
        Distribution.logp = logp
        Distribution.integral = integral
        variables = list(tf.placeholder(config.dtype) for i in range(dimension))
        return tuple(variables)

    return Distribution(FakeDistribution)


def test_parse_bounds():
    # TODO(chrisburr) Check these using FakeDistribution instead
    # Ensure specifying lower and upper works as expecte
    assert _parse_bounds(None, None, None) == [Region(-np.inf, np.inf)]
    assert _parse_bounds(4.2, None, None) == [Region(4.2, np.inf)]
    assert _parse_bounds(None, 3.6, None) == [Region(-np.inf, 3.6)]
    assert _parse_bounds(-5.1, 3.6, None) == [Region(-5.1, 3.6)]

    # Ensure specifying the bounds results in a list of Region objects
    bounds = [Region(-42, 35), Region(42, 46)]
    assert _parse_bounds(None, None, bounds) == [Region(-42, 35), Region(42, 46)]

    bounds = [-34, 22, 29, 108]
    assert _parse_bounds(None, None, bounds) == [Region(-34, 22), Region(29, 108)]

    bounds = [(-19, -5), (6, 21)]
    assert _parse_bounds(None, None, bounds) == [Region(-19, -5), Region(6, 21)]

    assert isinstance(_parse_bounds(None, None, bounds)[0], Region)


@raises(DistributionError)
def test_using_lower_and_bounds():
    FakeDistribution = get_fake_distribution()
    with Model():
        FakeDistribution(lower=5, bounds=[5, np.inf])


@raises(DistributionError)
def test_using_upper_and_bounds():
    FakeDistribution = get_fake_distribution()
    with Model():
        FakeDistribution(upper=101, bounds=[np.inf, 101])


@raises(DistributionError)
def test_requiring_logp():
    FakeDistribution = get_fake_distribution(logp=None)
    with Model():
        FakeDistribution()


@raises(NotImplementedError)
def test_numeric_integral():
    FakeDistribution = get_fake_distribution(integral=None)
    with Model():
        FakeDistribution()


# def test_n_dimensional():
#     FakeDistribution = get_fake_distribution(dimension=3)
#     with Model():
#         variables = FakeDistribution()

#     assert len(variables) == 3
#     assert isinstance(variables[2], tf.Tensor)


# def test_multiple_bounds_for_n_dimensional():
#     pass

# def test_BaseDistribution_is_tensor():
#    sess = tf.Session()
#    with Model() as model:
#        X = BaseDistribution()
#        assert isinstance(X, tf.Tensor)
#        # In order to prevent it from failing
#        model.untrack_variable(X)


# def test_BaseDistribution_can_eval():
#    sess = tf.Session()
#    with Model() as model:
#        X = BaseDistribution()
#        # In order to prevent it from failing
#        model.untrack_variable(X)

#    assert(sess.run(42 * X, feed_dict={X: 1.0}) == 42.0)
