
__all__ = ['exponential_log', 'normal_log', 'mix2_log']

def inside(val, left, right):
    return tf.logical_and(tf.greater_equal(val, left), tf.less_equal(val, right))

def exponential_log(X, lambda_, left, right):
    with tf.name_scope('exponential_log') as scope:
        integ = 1 - tf.exp(-(right - left) * lambda_)
        return tf.select(inside(X, left, right), tf.log(tf.abs(lambda_ / integ)) - (X - left) * lambda_, tf.fill(tf.shape(X), TYPE(-np.inf)))

def normal_log(X, mu, sigma, left, right):
    with tf.name_scope('normal_log') as scope:
        val = tf.log(1 / (tf.constant(np.sqrt(2 * np.pi), dtype=TYPE) * sigma)) - tf.pow(X - mu, 2) / (tf.constant(2, dtype=TYPE) * tf.pow(sigma, 2))
        cdf = lambda lim: 0.5 * tf.erfc((mu - lim) / (tf.constant(np.sqrt(2)) * sigma))
        integ = cdf(right) - cdf(left)
        return tf.select(inside(X, left, right), tf.log(tf.exp(val) / integ), tf.fill(tf.shape(X), TYPE(-np.inf)))

def mix2_log(X, logpdf1, logpdf2, f1, pdf1_params, pdf2_params, left, right):
    with tf.name_scope('mix2_log'):
        x1 = tf.log(f1) + logpdf1(X, *pdf1_params, left, right)
        x2 = tf.log(1 - f1) + logpdf2(X, *pdf2_params, left, right)

        x_max = tf.maximum(x1, x2)
        return x_max + tf.log(tf.exp(x1 - x_max) + tf.exp(x2 - x_max))

