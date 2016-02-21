from collections import namedtuple

import tensorflow as tf
import numpy as np
from scipy.optimize import minimize

from . import config
from . import utilities


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
        # A list of observed variables.
        # These are set in the `model.observed()` method
        # If this is none, `model.observed()` has not been called yet
        self._observed = None
        # A dictionary from `tensorflow.placeholder`s representing the hidden (latent)
        # variables of the model to `tensorflow.Variables` carrying the current state
        # of the model
        self._hidden = None
        # The graph that we will eventually run with
        self.session = tf.Session(graph=tf.Graph())
        self.model_graph = tf.get_default_graph()

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
        return self

    def __exit__(self, e_type, e, tb):
        Model._current_model = None

        # Re-raise underlying exceptions
        if e_type is not None:
            raise

    def observed(self, *args):
        if Model._current_model == self:
            raise ModelError("Can't call `model.observed()` inside the model block")

        for arg in args:
            if not arg in self._description:
                raise ValueError("Argument {} is not known to the model".format(arg))

        self._observed = dict()
        with self.session.graph.as_default():
            for arg in args:
                dummy = tf.Variable(0)
                self._observed[arg] = dummy
            self.session.run(tf.initialize_variables(list(self._observed.values())))


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
                self._hidden[var] = tf.Variable(var.dtype.as_numpy_dtype(assign_dict[var]))
            self.session.run(tf.initialize_variables(list(self._hidden.values())))

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
        return { k: v for k, v in zip(keys, values) }


    def _rewrite_graph(self, transform):
        input_map = { k.name: v for k, v in transform.items() }

        with self.session.graph.as_default():
            tf.import_graph_def(
                    self.model_graph.as_graph_def(),
                    input_map=input_map,
                    # Avoid adding a path prefix
                    name='test',
            )

    def pdf(self, *args):
        # TODO(ibab) make sure that args all have the same shape

        if len(args) != len(self._observed):
            raise ValueError("Different number of arguments passed to `model.pdf()` than declared in `model.observed()`")

        full_dict = self._hidden.copy()

        logps = []
        for var, arg in zip(self._observed, args):
            logps.append(self._description[var].logp)
            converted = var.dtype.as_numpy_dtype(arg)
            full_dict[var] = converted
        pdf = tf.add_n(logps)

        self._rewrite_graph(full_dict)
        new_pdf = self.session.graph.get_tensor_by_name('test/' + pdf.name)
        return self.session.run(new_pdf)

    def nll(self, *args):
        feed_dict = self._prepare_model(args)
        return self.session.run(self._nll, feed_dict=feed_dict)

    def fit(self, *args):
        feed_dict = self._prepare_model(args)
        placeholders = list(map(self._map_old_new.__getitem__, self._hidden))
        inits = self.session.run(self._hidden)

        def objective(xs):
            self.assign({k: v for k, v in zip(placeholders, xs)})
            return self.session.run(self._nll, feed_dict=feed_dict)

        bounds = []
        for h in self._hidden:
            # Slightly move the bounds so that the edges are not included
            p = self._map_old_new[h]
            if hasattr(p, 'lower') and p.lower is not None:
                lower = p.lower + 1e-10
            else:
                lower = None
            if hasattr(p, 'upper') and p.upper is not None:
                upper = p.upper - 1e-10
            else:
                upper = None

            bounds.append((lower, upper))

        return minimize(objective, inits, bounds=bounds)


__all__ = [
    Model,
]
