from tensorprob import Model
from tensorprob.distributions import BaseDistribution
from nose.tools import raises
import tensorflow as tf


#def test_BaseDistribution_is_tensor():
#    sess = tf.Session()
#    with Model():
#        dist = BaseDistribution()
#        assert isinstance(dist, tf.Tensor)
#
#
#def test_BaseDistribution_can_eval():
#    sess = tf.Session()
#    with Model():
#        dist = BaseDistribution()
#    assert(sess.run(42 * dist, feed_dict={dist: 1.0}) == 42.0)
