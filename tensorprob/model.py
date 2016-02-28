from collections import namedtuple
import logging
logger = logging.getLogger('tensorprob')

import numpy as np
import tensorflow as tf

from . import config
from .utilities import classproperty, generate_name, is_finite


# Used to specify valid ranges for variables
Region = namedtuple('Region', ['lower', 'upper'])
# Used to describe a variable's role in the model
Description = namedtuple('Description', ['logp', 'integral', 'bounds'])


class ModelError(RuntimeError):
    pass


class Model(object):
    """The probabilistic graph."""
    _current_model = None

    def __init__(self, name=None):
        # The description of the model. This is a dictionary from all
        # `tensorflow.placeholder`s representing the random variables of the
        # model (defined by the user in the model block) to their `Description`s
        self._description = dict()
        # A dictionary mapping the `tensorflow.placeholder`s representing the
        # observed variables of the model to `tensorflow.Variables`
        # These are set in the `model.observed()` method
        # If this is none, `model.observed()` has not been called yet
        self._observed = None
        # A dictionary from `tensorflow.placeholder`s representing the hidden (latent)
        # variables of the model to `tensorflow.Variables` carrying the current state
        # of the model
        self._hidden = None
        self._setters = None
        # A dictionary mapping `tensorflow.placeholder`s of variables to new
        # `tensorflow.placeholder`s which have been substituted using combinators
        self._silently_replace = dict()
        # The graph that the user's model is originally constructed in
        self._model_graph = tf.Graph()
        # The session that we will eventually run with
        self.session = tf.Session(graph=tf.Graph())

        # Whether `model.initialize()` has been called
        self.initialized = False
        self.name = name or generate_name(self.__class__)

    @classproperty
    def current_model(self):
        if Model._current_model is None:
            raise ModelError("This can only be used inside a model environment")
        return Model._current_model

    def __enter__(self):
        if Model._current_model is not None:
            raise ModelError("Can't nest models within each other")
        Model._current_model = self
        self.graph_ctx = self._model_graph.as_default()
        self.graph_ctx.__enter__()
        return self

    def __exit__(self, e_type, e, tb):
        Model._current_model = None

        self.graph_ctx.__exit__(e_type, e, tb)
        self.graph_ctx = None

        # Normalise all log probabilities contained in _description
        with self._model_graph.as_default():
            for var, (logp, integral, bounds) in self._description.items():
                logp -= tf.log(tf.add_n([integral(l, u) for l, u in bounds]))
                self._description[var] = Description(logp, integral, bounds)

        # We shouldn't be allowed to edit this one anymore
        self._model_graph.finalize()

        # Re-raise underlying exceptions
        if e_type is not None:
            raise

    def observed(self, *args):
        if Model._current_model == self:
            raise ModelError("Can't call `model.observed()` inside the model block")

        for arg in args:
            if arg not in self._description:
                raise ValueError("Argument {} is not known to the model".format(arg))

        self._observed = dict()
        self._setters = dict()
        with self.session.graph.as_default():
            for arg in args:
                dummy = tf.Variable(arg.dtype.as_numpy_dtype())
                self._observed[arg] = dummy
                setter_var = tf.Variable(arg.dtype.as_numpy_dtype(), name=arg.name.split(':')[0])
                setter = tf.assign(dummy, setter_var, validate_shape=False)
                self._setters[dummy] = (setter, setter_var)

    def _rewrite_graph(self, transform):
        input_map = {k.name: v for k, v in transform.items()}

        # Modify the input dictionary to replace variables which have been
        # superseded with the use of combinators
        for k, v in self._silently_replace.items():
            input_map[k.name] = self._observed[v]

        with self.session.graph.as_default():
            try:
                tf.import_graph_def(
                        self._model_graph.as_graph_def(),
                        input_map=input_map,
                        name='added',
                )
            except ValueError:
                # Ignore errors that ocour when the input_map tries to
                # rewrite a variable that isn't present in the graph
                pass

    def _get_rewritten(self, tensor):
        return self.session.graph.get_tensor_by_name('added/' + tensor.name)

    def initialize(self, assign_dict):
        # This is where the `self._hidden` map is created.
        # The `tensorflow.Variable`s of the map are initialized
        # to the values given by the user in `assign_dict`.

        if Model._current_model == self:
            raise ModelError("Can't call `model.initialize()` inside the model block")

        if self._observed is None:
            raise ModelError("Can't initialize latent variables before `model.observed()` has been called.")

        if self._hidden is not None:
            raise ModelError("Can't call `model.initialize()` twice. Use `model.assign()` to change the state.")

        if not isinstance(assign_dict, dict) or not assign_dict:
            raise ValueError("Argument to `model.initialize()` must be a dictionary with more than one element")

        for key in assign_dict.keys():
            if not isinstance(key, tf.Tensor):
                raise ValueError("Key in the initialization dict is not a tf.Tensor: {}".format(repr(key)))

        hidden = set(self._description.keys()).difference(set(self._observed))
        if hidden != set(assign_dict.keys()):
            raise ModelError("Not all latent variables have been passed in a call to `model.initialize().\n\
                    Missing variables: {}".format(hidden.difference(assign_dict.keys())))

        # Add variables to the execution graph
        with self.session.graph.as_default():
            self._hidden = dict()
            for var in hidden:
                self._hidden[var] = tf.Variable(var.dtype.as_numpy_dtype(assign_dict[var]), name=var.name.split(':')[0])
        self.session.run(tf.initialize_variables(list(self._hidden.values())))
        # Sort the hidden variables so we can access them in a consistant order
        self._hidden_sorted = sorted(self._hidden.keys(), key=lambda v: v.name)
        for h in self._hidden.values():
            with self.session.graph.as_default():
                var = tf.Variable(h.dtype.as_numpy_dtype(), name=h.name.split(':')[0] + '_placeholder')
                setter = h.assign(var)
            self._setters[h] = (setter, var)

        all_vars = self._hidden.copy()
        all_vars.update(self._observed)

        self._rewrite_graph(all_vars)

        with self.session.graph.as_default():
            # observed_logps contains one element per data point
            observed_logps = [self._get_rewritten(self._description[v].logp) for v in self._observed]
            # hidden_logps contains a single value
            hidden_logps = [self._get_rewritten(self._description[v].logp) for v in self._hidden]

            # Handle the case where we don't have observed variables.
            # We define the probability to not observe anything as 1.
            if not observed_logps:
                observed_logps = [tf.constant(0, dtype=config.dtype)]

            self._pdf = tf.exp(tf.add_n(
                observed_logps
            ))
            self._nll = -tf.add_n(
                [tf.reduce_sum(logp) for logp in observed_logps] +
                hidden_logps
            )

            variables = [self._hidden[k] for k in self._hidden_sorted]
            self._nll_grad = tf.gradients(self._nll, variables)
            for i, (v, g) in enumerate(zip(variables, self._nll_grad)):
                if g is None:
                    self._nll_grad[i] = tf.constant(0, dtype=config.dtype)
                    logger.warn('Model is independent of variable {}'.format(
                        v.name.split(':')[0]
                    ))

        self.initialized = True

    def assign(self, assign_dict):
        if Model._current_model == self:
            raise ModelError("Can't call `model.assign()` inside the model block")

        if not isinstance(assign_dict, dict) or not assign_dict:
            raise ValueError("Argument to assign must be a dictionary with more than one element")

        if self._observed is None:
            raise ModelError("Can't assign state to the model before `model.observed()` has been called.")

        if self._hidden is None:
            raise ModelError("Can't assign state to the model before `model.initialize()` has been called.")

        # Assign values without adding to the graph
        setters = [ self._setters[self._hidden[k]][0] for k, v in assign_dict.items() ]
        feed_dict = { self._setters[self._hidden[k]][1]: v for k, v in assign_dict.items() }
        for s in setters:
            self.session.run(s, feed_dict=feed_dict)

    @property
    def state(self):
        keys = self._hidden.keys()
        variables = list(self._hidden.values())
        values = self.session.run(variables)
        return {k: v for k, v in zip(keys, values)}

    def _set_data(self, data):
        if self._hidden is None:
            raise ModelError("Can't use the model before it has been initialized with `model.initialize(...)`")
        # TODO(ibab) make sure that args all have the correct shape
        if len(data) != len(self._observed):
            raise ValueError("Different number of arguments passed to model method than declared in `model.observed()`")

        ops = []
        feed_dict = {self._setters[k][1]: v for k, v in zip(self._observed.values(), data)}
        for obs, arg in zip(self._observed.values(), data):
            ops.append(self._setters[obs][0])
        for s in ops:
            self.session.run(s, feed_dict=feed_dict)

    def pdf(self, *args):
        self._set_data(args)
        return self.session.run(self._pdf)

    def nll(self, *args):
        self._set_data(args)
        return self.session.run(self._nll)

    def fit(self, *args, **kwargs):
        optimizer = kwargs.get('optimizer')
        use_gradient = kwargs.get('use_gradient', True)
        self._set_data(args)

        variables = [self._hidden[k] for k in self._hidden_sorted]
        gradient = self._nll_grad if use_gradient else None

        # Some optimizers need bounds
        bounds = []
        for h in self._hidden_sorted:
            # Take outer bounds into account.
            # We can't do better than that here
            lower = self._description[h].bounds[0].lower
            upper = self._description[h].bounds[-1].upper
            bounds.append((lower, upper))

        if optimizer is None:
            from .optimizers import ScipyLBFGSBOptimizer
            optimizer = ScipyLBFGSBOptimizer()

        optimizer.session = self.session

        out = optimizer.minimize(variables, self._nll, gradient=gradient, bounds=bounds)
        self.assign({k: v for k, v in zip(sorted(self._hidden.keys(), key=lambda x: x.name), out.x)})
        return out

    def mcmc(self, *args, **kwargs):
        sampler = kwargs.get('sampler')
        samples = kwargs.get('samples')
        self._set_data(args)

        if sampler is None:
            from .samplers import EmceeSampler
            sampler = EmceeSampler(walkers=40, session=self.session)

        return sampler.sample(list(self._hidden.values()), self._nll, samples=samples)

__all__ = [
    Model,
]
