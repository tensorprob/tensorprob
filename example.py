import tensorprob as tp

model = tp.Model()

with model:
    mu = tp.scalar('mu')
    f_normal = tp.scalar('f_normal', lower=0, upper=1)
    sigma1 = tp.scalar('sigma1', lower=0)
    sigma2 = tp.scalar('sigma2', lower=0)
    lamb = tp.scalar('lambda')
    f = tp.scalar('f', lower=0, upper=1)

    X1 = tp.Normal2(mu, f, sigma1, sigma2)
    X2 = tp.Exponential(lamb)
    X_ = tp.Mix2(f, X1, X2)

    X = tp.Bound(X_, 100, 200)

inits = {mu: 150, sigma1: 10, sigma2: 20, f_normal: 0.5, lamb: 0.01, f: 0.2, }

model.fit(X=[1, 2, 3], inits=inits)
