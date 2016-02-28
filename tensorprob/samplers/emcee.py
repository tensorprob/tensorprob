from __future__ import absolute_import

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

        def objective(xs):
            feed_dict = { k: v for k, v in zip(variables, xs) }
            out = self.session.run(cost, feed_dict=feed_dict)
            if np.isnan(out):
                return np.inf
            return -out

        all_inits = self.emcee.utils.sample_ball(inits, [1e-1] * len(inits), self.walkers)
        sampler = self.emcee.EnsembleSampler(self.walkers, len(variables), objective)

        samples = 1 if samples is None else samples
        sampler.random_state = np.random.mtrand.RandomState(np.random.randint(1)).get_state()
        pos, lnprob, rstate = sampler.run_mcmc(all_inits, samples)
        return sampler.chain

