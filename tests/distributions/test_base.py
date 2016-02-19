import tensorflow as tf
from tensorprob.distributions import BaseDistribution
from nose.tools import raises

@raises(tf.python.framework.errors.InvalidArgumentError)
def test_BaseDistribution_is_placeholder():
    dist = BaseDistribution()
    sess = tf.Session()
    sess.run(42 * dist)

def test_BaseDistribution_can_eval():
    dist = BaseDistribution()
    sess = tf.Session()
    assert(sess.run(42 * dist, feed_dict={dist: 1.0}) == 42.0)

