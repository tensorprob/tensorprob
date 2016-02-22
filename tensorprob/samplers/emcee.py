import tensorflow as tf
import numpy as np

from .base import BaseSampler

class EmceeSampler(BaseSampler):

    def __init__(self, walkers=100, **kwargs):
        try:
            import emcee
        except ImportError:
            raise ImportError("The emcee package needs to be installed in order to use EmceeSampler")

        self.emcee = emcee
        self.walkers = walkers
        super(EmceeSampler, self).__init__(**kwargs)

    def sample(self, variables, cost, gradient=None, samples=None):
        # Check if variables is iterable
        try:
            iter(variables)
        except TypeError:
            raise ValueError("Variables parameter is not iterable")

        inits = self.session.run(variables)

        for v in variables:
            if not isinstance(v, tf.Variable):
                raise ValueError("Parameter {} is not a tensorflow variable".format(v))

        self.fev = 0
        def objective(xs):
            feed_dict = { k: v for k, v in zip(variables, xs) }
            out = self.session.run(cost, feed_dict=feed_dict)
            if self.fev % 1000 == 0:
                print(self.fev)
            self.fev += 1
            print(out)
            if np.isnan(out):
                return np.inf
            return -out

        all_inits = self.emcee.utils.sample_ball(inits, [1e-2] * len(inits), self.walkers)
        sampler = self.emcee.EnsembleSampler(self.walkers, len(variables), objective)

        samples = 1 if samples is None else samples
        pos, lnprob, rstate = sampler.run_mcmc(all_inits, samples)
        return sampler.flatchain

