
from tensorprob import ScipyLBFGSBOptimizer, MigradOptimizer
import numpy as np
import tensorflow as tf
from nose.tools import raises

def test_scipy_lbfgsb():
    sess = tf.Session()
    x = tf.Variable(np.float64(2), name='x')
    sess.run(tf.initialize_variables([x]))
    optimizer = ScipyLBFGSBOptimizer(verbose=True, session=sess)
    # With gradient
    results = optimizer.minimize([x], x**2, [2 * x])
    assert results.success
    # Without gradient
    results = optimizer.minimize([x], x**2)
    assert results.success
    # Test callback
    def callback(xs):
        pass
    optimizer = ScipyLBFGSBOptimizer(verbose=True, session=sess, callback=callback)
    assert optimizer.minimize([x], x**2).success
    @raises(ValueError)
    def test_illegal_parameter_as_variable1():
        optimizer.minimize([42], x**2)
    test_illegal_parameter_as_variable1()
    @raises(ValueError)
    def test_illegal_parameter_as_variable2():
        optimizer.minimize(42, x**2)
    test_illegal_parameter_as_variable2()

def test_migrad():
    sess = tf.Session()
    x = tf.Variable(np.float64(2), name='x')
    sess.run(tf.initialize_variables([x]))
    optimizer = MigradOptimizer(session=sess)
    # With gradient
    results = optimizer.minimize([x], x**2, [2 * x])
    assert results.success
    # Without gradient
    results = optimizer.minimize([x], x**2)
    assert results.success
    @raises(ValueError)
    def test_illegal_parameter_as_variable1():
        optimizer.minimize([42], x**2)
    test_illegal_parameter_as_variable1()
    @raises(ValueError)
    def test_illegal_parameter_as_variable2():
        optimizer.minimize(42, x**2)
    test_illegal_parameter_as_variable2()

