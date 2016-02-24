from collections import namedtuple
import logging
logger = logging.getLogger(__name__)

import tensorflow as tf

from . import utilities
from . import config


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
        # A dictionary mapping `tensorflow.placeholder`s of variables to new
        # `tensorflow.placeholder`s which have been substituted using combinators
        self._silently_replace = dict()
        # The graph that the user's model is originally constructed in
        self._model_graph = tf.Graph()
        # The session that we will eventually run with
        self.session = tf.Session(graph=tf.Graph())

        # Whether `model.initialize()` has been called
        self.initialized = False
        self.name = name or utilities.generate_name(self.__class__)

    @utilities.classproperty
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
        with self.session.graph.as_default():
            for arg in args:
                dummy = tf.Variable(arg.dtype.as_numpy_dtype())
                self._observed[arg] = dummy

    def _rewrite_graph(self, transform):
        input_map = {k.name: v for k, v in transform.items()}

        # Modify the input dictionary to replace variables which have been
        # superseded with the use of combinators
        for k, v in self._silently_replace.items():
            if v.name in input_map:
                del input_map[v.name]
            input_map[k.name] = self._observed[v]

        with self.session.graph.as_default():
            try:
                tf.import_graph_def(
                        self._model_graph.as_graph_def(),
                        input_map=input_map,
                        name='added',
                )
            except:
                # Handle the case where there is no likelihood
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

        all_vars = self._hidden.copy()
        all_vars.update(self._observed)

        self._rewrite_graph(all_vars)

        with self.session.graph.as_default():
            logps = []
            for var in self._observed:
                logps.append(self._get_rewritten(self._description[var].logp))

            self._pdf = tf.exp(tf.add_n(logps))
            self._nll = -tf.add_n([tf.reduce_sum(logp) for logp in logps])
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

        ops = list()
        for k, v in assign_dict.items():
            ops.append(self._hidden[k].assign(v))
        self.session.run(tf.group(*ops))

    @property
    def state(self):
        keys = self._hidden.keys()
        variables = list(self._hidden.values())
        values = self.session.run(variables)
        return {k: v for k, v in zip(keys, values)}

    def _set_data(self, data):
        if not self.initialized:
            raise ModelError("Can't use the model before it has been initialized with `model.initialize(...)`")
        # TODO(ibab) make sure that args all have the correct shape
        if len(data) != len(self._observed):
            raise ValueError("Different number of arguments passed to model method than declared in `model.observed()`")

        ops = []
        for obs, arg in zip(self._observed.values(), data):
            ops.append(tf.assign(obs, obs.dtype.as_numpy_dtype(arg), validate_shape=False))
        self.session.run(tf.group(*ops))

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

        return optimizer.minimize(variables, self._nll, gradient=gradient, bounds=bounds)


__all__ = [
    Model,
]
