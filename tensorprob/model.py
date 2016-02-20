import tensorflow as tf

from . import utilities
from .distributions import BaseDistribution

from scipy.optimize import minimize

class ModelError(RuntimeError):
    pass


class Model:
    """The probabilistic graph."""
    _current_model = None

    def __init__(self, name=None):
        self._logp = tf.constant(0)
        self._components = []
        self._observed = None
        self.name = name or utilities.generate_name()
        self.session = tf.Session()

    def __enter__(self):
        if Model._current_model is not None:
            raise ModelError("Can't nest models within each other")
        Model._current_model = self
        return self

    def __exit__(self, e_type, e, tb):
        Model._current_model = None
        self.session.run(tf.initialize_all_variables())

    def track_variable(self, obj):
        """Add *obj* to the list of tracked objects."""
        self._components.append(obj)

    def untrack_variable(self, obj):
        """Remove *obj* to the list of tracked objects."""
        self._components.remove(obj)

    def pdf(self, *args):
        feed_dict = self._prepare_model(args)
        return self.session.run(tf.exp(self._logp), feed_dict=feed_dict)

    def logp(self, *args):
        feed_dict = self._prepare_model(args)
        return self.session.run(self._logp, feed_dict=feed_dict)

    def nll(self, *args):
        feed_dict = self._prepare_model(args)
        nll = -tf.reduce_sum(self._logp)
        return self.session.run(nll, feed_dict=feed_dict)

    def fit(self, *args):
        feed_dict = self._prepare_model(args)
        params = self.parameters
        nll = -tf.reduce_sum(self._logp)
        inits = self.session.run(params)

        def objective(xs):
            self.assign({ k: v for k, v in zip(params, xs)})
            return self.session.run(nll, feed_dict=feed_dict)

        bounds = []
        for p in params:
            # Slightly move the bounds so that the edges are not included
            bounds.append((p.lower + 1e-10, p.upper - 1e-10))

        return minimize(objective, inits, bounds=bounds)

    def _prepare_model(self, args):
        if self._observed is None:
            raise ModelError("observed() has not been called")

        if len(args) != len(self._observed):
            raise ModelError("Number of parameters does not correspond to observed variables")

        logps = []
        for c in self.components:
            if isinstance(c, BaseDistribution):
                logps.append(c.logp())

        with tf.name_scope(self.name):
            self._logp = tf.add_n(logps)

        feed_dict = dict()
        for obs, arg in zip(self._observed, args):
            feed_dict[obs] = arg

        return feed_dict

    def assign(self, assign_dict):
        ops = [k.assign(v) for k, v in assign_dict.items()]
        self.session.run(tf.group(*ops))

    @property
    def components(self):
        return self._components

    @property
    def parameters(self):
        return [x for x in self._components if x not in self._observed]

    def observed(self, *args):
        for arg in args:
            if not isinstance(arg, BaseDistribution):
                raise ValueError("Argument {} is not a distribution".format(arg))
        self._observed = args

    @utilities.classproperty
    def current_model(self):
        if Model._current_model is None:
            raise ModelError("This can only be used inside a model environment")
        return Model._current_model


__all__ = [
    Model,
]
