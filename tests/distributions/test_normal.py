import tensorprob as tp


def make_normal():
    mu = tp.Scalar('mu')
    sigma = tp.Scalar('sigma', lower=0)
    distribution = tp.Normal(mu, sigma)
    return mu, sigma, distribution


def test_init():
    mu, sigma, distribution = make_normal()
    assert(distribution.mu is mu)
    assert(distribution.sigma is sigma)


def test_pdf():
    pass
