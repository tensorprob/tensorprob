import tensorprob as tp

x_data = [1, 2, 3]

with tp.Model() as m:
    mu = Parameter('mu')
    f_normal = Parameter('f_normal', lower=0, upper=1)
    sigma1 = Parameter('sigma1', lower=0)
    sigma2 = Parameter('sigma2', lower=0)
    lamb = Parameter('lambda')
    f = Parameter('f', lower=0, upper=1)

    X1 = Normal2(mu, f_normal, sigma1, sigma2)
    X2 = Exponential(lamb)
    X_ = Mix2(f, X1, X2)

    X = Bound(X_, 100, 200)

    observed(X)

set_values({
    mu: 150,
    sigma1: 10,
    sigma2: 20,
    f_normal: 0.5,
    lamb: 0.01,
    f: 0.2
})

fit(x_data)
