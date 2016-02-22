from tensorprob import Model
from nose.tools import raises
import tensorflow as tf


#def test_BaseDistribution_is_tensor():
#    sess = tf.Session()
#    with Model() as model:
#        X = BaseDistribution()
#        assert isinstance(X, tf.Tensor)
#        # In order to prevent it from failing
#        model.untrack_variable(X)
#
#
#def test_BaseDistribution_can_eval():
#    sess = tf.Session()
#    with Model() as model:
#        X = BaseDistribution()
#        # In order to prevent it from failing
#        model.untrack_variable(X)
#
#    assert(sess.run(42 * X, feed_dict={X: 1.0}) == 42.0)
