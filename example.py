import tensorprob as tp


with tp.Model() as m:
    mu = m.Scalar('mu')
    f_normal = m.Scalar('f_normal', lower=0, upper=1)
    sigma1 = m.Scalar('sigma1', lower=0)
    sigma2 = m.Scalar('sigma2', lower=0)
    lamb = m.Scalar('lambda')
    f = m.Scalar('f', lower=0, upper=1)

    X1 = m.Normal2(mu, f_normal, sigma1, sigma2)
    X2 = m.Exponential(lamb)
    X_ = m.Mix2(f, X1, X2)

    X = m.Bound(X_, 100, 200)

inits = {mu: 150, sigma1: 10, sigma2: 20, f_normal: 0.5, lamb: 0.01, f: 0.2, }

m.fit(X=[1, 2, 3], inits=inits)
