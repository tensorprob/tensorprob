from collections import namedtuple

import tensorflow as tf
from scipy.optimize import minimize

from . import config
from . import utilities


Description = namedtuple('Description', ['logp', 'integral', 'bounds'])


class ModelError(RuntimeError):
    pass


class Model(object):
    """The probabilistic graph."""
    _current_model = None

    def __init__(self, name=None, session=None):
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

        self.name = name or utilities.generate_name(self.__class__)
        self.session = session or tf.Session()

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

        #logps = set(c.logp for c in self._components)
        ## Don't fail with empty models
        #if self._components:
        #    with tf.name_scope(self.name):
        #        summed = tf.add_n(list(map(tf.reduce_sum, logps)))
        #        self._original_nll = -summed

    def observed(self, *args):
        if Model._current_model == self:
            raise ModelError("Can't call `model.observed()` inside the model block")

        for arg in args:
            if not arg in self._description:
                raise ValueError("Argument {} is not known to the model".format(arg))

        self._observed = args

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

        hidden = set(self._description.keys().difference(set(self._observed)))
        if hidden != set(assign_dict.keys()):
            raise ModelError("Not all latent variables have been passed in a call to `model.initialize().\n\
                    Missing variables: {}".format(hidden.difference(assign_dict.keys())))

        self._hidden = dict()
        for var in hidden:
            self._hidden[var] = tf.Variable(assign_dict[var])

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
        values = self._hidden.values()

        hidden_state = self.session.run(values)
        return { k: v for k, v in zip(keys, hidden_state) }

    def _prepare_model(self, args):
        if self._observed is None:
            raise ModelError("observed() has not been called")

        if len(args) != len(self._observed):
            raise ModelError(
                "Number of parameters ({0}) does not correspond to observed "
                "variables ({1})".format(len(args), len(self._observed))
            )

        feed_dict = dict()
        for obs, arg in zip(self._observed_new, args):
            feed_dict[obs] = arg

        return feed_dict

        # Every node that's not observed is a hidden variable with state.
        # Rewrite the graph to convert the tf.placeholders for these into tf.Variables.
        # Currently, we assume that all hidden variables are scalars, because we're lazy.
        # TODO(ibab) allow hidden variables to be of any tensor shape.
        hidden = []
        for x in self._components:
            if not x in args:
                hidden.append(x)

        # Use the original graph_def defined in the model block as the basis for the rewrite
        original = self._original_graph_def

        self._hidden = []
        self._observed = args
        self._observed_new = []
        observable_logps_old = []
        self._map_old_new = dict()

        with tf.Graph().as_default() as g:

            input_map = dict()
            for a in args:
                tmp = tf.placeholder(a.dtype, name=a.name.split(':')[0])
                self._observed_new.append(tmp)
                observable_logps_old.append(a.logp())
                self._map_old_new[a] = tmp
                self._map_old_new[tmp] = a
                input_map[a.name] = tmp
            for h in hidden:
                var = tf.Variable(config.dtype(0), name=h.name.split(':')[0])
                self._hidden.append(var)
                self._map_old_new[h] = var
                self._map_old_new[var] = h
                input_map[h.name] = var

            tf.import_graph_def(
                    original,
                    input_map=input_map,
                    # Avoid adding a path prefix
                    name='',
            )

            # We set these to versions that use the tf.Variables internally
            self._nll = g.get_tensor_by_name(self._original_nll.name)

            observable_logps_new = []
            for ol in observable_logps_old:
                observable_logps_new.append(g.get_tensor_by_name(ol.name))

            self._observable_pdf = tf.exp(tf.add_n(observable_logps_new))

            # We override this session's graph with g
            # TODO(ibab) find a nicer way to do this (might require changing
            # things upstream)
            self.session._graph = g
            self.session.run(tf.initialize_all_variables())


    def pdf(self, *args):
        # TODO(ibab) make sure that args all have the same shape

        if len(args) != len(self._observed):
            raise ValueError("Different number of arguments passed to `model.pdf()` than declared in `model.observed()`")

        full_dict = self._hidden.copy()

        logps = []
        for var, arg in zip(self._observed, args):
            logps.append(self._description[var].logp)
            full_dict[var] = tf.Variable(arg, name=var.name.split(':')[0])
        pdf = tf.add_n(logps)

        new_graph = self._rewrite_graph(full_dict)
        with tf.Session(graph=new_graph):
            out = self.session.run(pdf)
        return out

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
