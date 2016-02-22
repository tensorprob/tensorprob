
from tensorprob import ScipyLBFGSBOptimizer
import tensorflow as tf

def test_scipy_lbfgsb():
    sess = tf.Session()
    x = tf.Variable(2, name='x')
    sess.run(tf.initialize_variables([x]))
    optimizer = ScipyLBFGSBOptimizer(session=sess)
    optimizer.minimize([x], x**2)

