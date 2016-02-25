import numpy as np
import scipy.stats as st
from nose.tools import raises
from numpy.testing import assert_array_almost_equal

from tensorprob import (
    config,
    Distribution,
    DistributionError,
    Model,
    ModelError,
    Normal,
    Parameter,
    Region,
)
import tensorflow as tf


def get_fake_distribution(logp=-42, integral=-42, dimension=1):
    if integral == -42:
        def integral(lower, upper):
            return tf.constant(2, dtype=config.dtype)

    def FakeDistribution(name=None):
        FakeDistribution.name = name

        if logp == -42:
            Distribution.logp = tf.constant(1, dtype=config.dtype)
        else:
            Distribution.logp = logp

        Distribution.integral = integral
        variables = tuple(tf.placeholder(config.dtype) for i in range(dimension))
        return variables

    return Distribution(FakeDistribution)


@raises(ModelError)
def test_inside_another_graph():
    FakeDistribution = get_fake_distribution()
    other_sessions = tf.Session()
    with Model():
        with other_sessions.graph.as_default():
            FakeDistribution()


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


@raises(ValueError)
def test_bounds_invalid_odd():
    FakeDistribution = get_fake_distribution()
    with Model():
        FakeDistribution(bounds=[5.5, 6.6, 7.7, 8.8, 9.9])


@raises(ValueError)
def test_bounds_invalid_shape_1():
    FakeDistribution = get_fake_distribution()
    with Model():
        FakeDistribution(bounds=[[1.1, 2.2, 3.3]])


@raises(ValueError)
def test_bounds_invalid_shape_2():
    FakeDistribution2D = get_fake_distribution(dimension=2)
    with Model():
        FakeDistribution2D(bounds=[
            [1.1, 2.2, 3.3, 4.4],
            [1.1, 2.2, 3.3, 4.4],
            [1.1, 2.2, 3.3, 4.4]
        ])


@raises(ValueError)
def test_bounds_contain_none():
    FakeDistribution2D = get_fake_distribution(dimension=2)
    with Model():
        FakeDistribution2D(bounds=[
            [1.1, 2.2, 3.3, 4.4],
            [1.1, None, 3.3, 4.4],
            [1.1, 2.2, 3.3, 4.4]
        ])


def test_bounds_1D():
    FakeDistribution = get_fake_distribution()
    with Model() as model:
        A = FakeDistribution()
        B = FakeDistribution(lower=1.1)
        C = FakeDistribution(upper=2.2)
        D = FakeDistribution(lower=3.3, upper=4.4)
        E = FakeDistribution(bounds=[(5.5, 6.6), (7.7, 8.8), (9.9, 11.11)])

    assert model._description[A].bounds == [Region(-np.inf, np.inf)]
    assert model._description[B].bounds == [Region(1.1, np.inf)]
    assert model._description[C].bounds == [Region(-np.inf, 2.2)]
    assert model._description[D].bounds == [Region(3.3, 4.4)]
    assert model._description[E].bounds == [Region(5.5, 6.6), Region(7.7, 8.8), Region(9.9, 11.11)]


def test_bounds_ND():
    FakeDistribution3D = get_fake_distribution(dimension=3)
    with Model() as model:
        A1, A2, A3 = FakeDistribution3D()
        B1, B2, B3 = FakeDistribution3D(lower=1.1)
        C1, C2, C3 = FakeDistribution3D(upper=2.2)
        D1, D2, D3 = FakeDistribution3D(lower=3.3, upper=4.4)
        E1, E2, E3 = FakeDistribution3D(bounds=[(5.5, 6.6), (7.7, 8.8), (9.9, 11.1)])
        F1, F2, F3 = FakeDistribution3D(bounds=[
            [(1.1, 2.1), (3.1, 4.1), (5.1, 6.1)],
            [(1.2, 2.2), (3.2, 4.2), (5.2, 6.2)],
            [(1.3, 2.3), (3.3, 4.3), (5.3, 6.3)],
        ])

    assert model._description[A1].bounds == [Region(-np.inf, np.inf)]
    assert model._description[A2].bounds == [Region(-np.inf, np.inf)]
    assert model._description[A3].bounds == [Region(-np.inf, np.inf)]

    assert model._description[B1].bounds == [Region(1.1, np.inf)]
    assert model._description[B2].bounds == [Region(1.1, np.inf)]
    assert model._description[B3].bounds == [Region(1.1, np.inf)]

    assert model._description[C1].bounds == [Region(-np.inf, 2.2)]
    assert model._description[C2].bounds == [Region(-np.inf, 2.2)]
    assert model._description[C3].bounds == [Region(-np.inf, 2.2)]

    assert model._description[D1].bounds == [Region(3.3, 4.4)]
    assert model._description[D2].bounds == [Region(3.3, 4.4)]
    assert model._description[D3].bounds == [Region(3.3, 4.4)]

    assert model._description[E1].bounds == [Region(5.5, 6.6), Region(7.7, 8.8), Region(9.9, 11.1)]
    assert model._description[E2].bounds == [Region(5.5, 6.6), Region(7.7, 8.8), Region(9.9, 11.1)]
    assert model._description[E3].bounds == [Region(5.5, 6.6), Region(7.7, 8.8), Region(9.9, 11.1)]

    assert model._description[F1].bounds == [Region(1.1, 2.1), Region(3.1, 4.1), Region(5.1, 6.1)]
    assert model._description[F2].bounds == [Region(1.2, 2.2), Region(3.2, 4.2), Region(5.2, 6.2)]
    assert model._description[F3].bounds == [Region(1.3, 2.3), Region(3.3, 4.3), Region(5.3, 6.3)]


def test_integral():
    bounds = [
        (-1, -0.9), (-0.65, -0.5), (-0.4, -0.32),
        (-0.31, -0.1), (0, 0.05), (0.2, np.inf)
    ]

    @np.vectorize
    def allowed_point(x):
        for l, u in bounds:
            if l < x and x < u:
                return 1
        return 0

    with Model() as model:
        mu = Parameter()
        sigma = Parameter(lower=0)
        X = Normal(mu, sigma, bounds=bounds)

    model.observed(X)
    model.initialize({mu: -0.4, sigma: 2})

    xs = np.linspace(-1, 1, 1001)

    # Calculate the integral using scipy, zeroing points that are out of bounds
    out1 = st.norm.pdf(xs, -0.4, 2) * allowed_point(xs)
    integral = sum(st.norm.cdf(u, -0.4, 2) - st.norm.cdf(l, -0.4, 2) for l, u in bounds)
    out1 /= integral

    out2 = model.pdf(xs)

    assert_array_almost_equal(out1, out2, 15)
