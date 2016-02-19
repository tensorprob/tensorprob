import tensorflow as tf

from . import distributions
from .scalar import Scalar


def get_current_model():
    return Model.__current_model


class Model:
    '''
    The probabilistic graph.
    '''
    __current_model = None

    def __init__(self, random_state=None):
        self._logp = tf.constant(0)
        self.components = []

        self.Scalar = self.init_object(Scalar)
        for distribution in distributions.__all__:
            if distribution is not distributions.BaseDistribution:
                setattr(self, distribution.__name__, self.init_object(distribution))

    def __enter__(self):
        if Model.__current_model is not None:
            raise ValueError("Can't nest models within each other")
        Model.__current_model = self
        return self

    def __exit__(self, e_type, e, tb):
        Model.__current_model = None

    def init_object(self, cls):
        def init(*args, **kwargs):
            obj = cls(*args, **kwargs)
            self.track_object(obj)
            return obj
        return init

    def track_object(self, obj):
        self.components.append(obj)

    def pdf(self, **kwargs):
        pass

    def logp(self, **kwargs):
        pass

    def nodes(self):
        pass

    def parameters(self):
        pass

    def increment_logp(self, logp):
        if Model.__current_model != self:
            raise ValueError(
                "Can't increment logp for this model, as it is not the current one"
            )

        self._logp += logp


__all__ = [
    Model,
    get_current_model
]
