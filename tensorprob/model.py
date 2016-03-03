from collections import OrderedDict
import logging
logger = logging.getLogger('tensorprob')

import numpy as np
import tensorflow as tf

from . import config
from .utilities import (
    classproperty,
    Description,
    generate_name,
    ModelSubComponet,
    Region,
    set_logp_to_neg_inf
)


class ModelError(RuntimeError):
    pass


class Model(object):
    '''The model class is the primary interface of TensorProb. It allows you to
    declare random variables, describe the (directed) probabalistic
    relationships between them, provide observations for some of them, and
    perform inference on the unobserved (latent) variables.

    Models are agnostic as to whether you want to follow frequentist or
    bayesian paradigms of inference. They allow you to find the maximum
    likelihood or maximum a posteriori estimate for you model given the data
    using the `.fit` method, or to sample from the likelihood/posterior using
    MCMC techniques (See the `.mcmc` method).

    Random variables can only be instantiated inside the `with` context of a model,
    and each model can only have a single `with` block.

    Inside the `with` context of the model, you can define variables and their
    relationships by telling a "generative story".
    For example, defining a new variable `X` with `X ~ Normal(0, 1)` is written as
    `X = Normal(0, 1)`.
    Random variables can then be plugged in as the conditional parameters of
    other distributions.

    After the `.initialize` method is called, the model has a *state* for each latent
    variable, which is used for the initial parameters in the `.fit` and `.mcmc` methods,
    as well as when using the `.pdf` method.

    Parameters
    ----------
    name : string, default None
        An optional name for this model. This is currently not used, but
        should be useful when working with multiple models simultaneously in
        the future.

    Examples
    --------
    >>> with Model() as model:
    ...     n = Parameter(lower=0)
    ...     N = Poisson(n)
    ... model.observed(N)
    ... model.initialize({ n: 10 })
    ... model.fit([20])
    '''

    _current_model = None

    def __init__(self, name=None):
        # The description of the model. This is a dictionary from all
        # `tensorflow.placeholder`s representing the random variables of the
        # model (defined by the user in the model block) to their `Description`s
        self._description = dict()
        self._full_description = dict()
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
        '''Returns the currently active `Model` when inside its `with` block.'''
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

        # Normalise all log probabilities contained in _description
        for var, (logp, integral, bounds, frac, _) in self._full_description.items():
            logp -= tf.log(tf.add_n([integral(l, u) for l, u in bounds]))

            # Force logp to negative infinity when outside the allowed bounds
            logp = set_logp_to_neg_inf(var, logp, bounds)

            # Add the changed logp to the model description
            self._full_description[var] = Description(logp, integral, bounds, frac, _)
            if var in self._description:
                self._description[var] = Description(logp, integral, bounds, frac, _)

        # Exit the tensorflow graph
        self.graph_ctx.__exit__(e_type, e, tb)
        self.graph_ctx = None

        # We shouldn't be allowed to edit this one anymore
        self._model_graph.finalize()

        # Re-raise underlying exceptions
        if e_type is not None:
            raise

    def __getitem__(self, key):
        if key not in self._full_description:
            raise KeyError

        logp, integral, bounds, frac, _ = self._full_description[key]

        def pdf(*args):
            self._set_data(args)
            return self.session.run(
                tf.exp(self._get_rewritten(logp)) *
                self._get_rewritten(frac)
            )

        return ModelSubComponet(pdf)

    def observed(self, *args):
        '''Declares the random variables in `args` as observed, which means
        that data is available for them.

        The order in which variables are used here defines the order in which
        they will have to be passed in later when using methods like `.fit` or
        `.mcmc`. All variables in the model that are not declared as observed
        are automatically declared as *latent* and become the subject of
        inference.

        `.observed` can only be called once per `Model` and is a requirement
        for calling `.initialize`.

        Parameters
        ----------
        *args : random variables
        The random variables for which data is available.
        '''
        if Model._current_model == self:
            raise ModelError("Can't call `model.observed()` inside the model block")

        for arg in args:
            if arg not in self._description:
                raise ValueError("Argument {} is not known to the model".format(arg))

        self._observed = OrderedDict()
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
        '''Allows you to specify the initial state of the unobserved (latent)
        variables.

        Can only be called after observed variables have been declared with
        `.observed`.

        Parameters
        ----------
        assign_dict : dict
            A dictionary from random variables to values.
            This has to specify a value for all unobserved (latent) variables
            of the model.
        '''
        # This is where the `self._hidden` map is created.
        # The `tensorflow.Variable`s of the map are initialized
        # to the values given by the user in `assign_dict`.

        if Model._current_model == self:
            raise ModelError("Can't call `model.initialize()` inside the model block")

        if self._observed is None:
            raise ModelError("Can't initialize latent variables before "
                             "`model.observed()` has been called.")

        if self._hidden is not None:
            raise ModelError("Can't call `model.initialize()` twice. Use "
                             "`model.assign()` to change the state.")

        if not isinstance(assign_dict, dict) or not assign_dict:
            raise ValueError("Argument to `model.initialize()` must be a "
                             "dictionary with more than one element")

        for key in assign_dict.keys():
            if not isinstance(key, tf.Tensor):
                raise ValueError("Key in the initialization dict is not a "
                                 "tf.Tensor: {}".format(repr(key)))

        hidden = set(self._description.keys()).difference(set(self._observed))
        if hidden != set(assign_dict.keys()):
            raise ModelError("Not all latent variables have been passed in a "
                             "call to `model.initialize().\nMissing variables: {}"
                             .format(hidden.difference(assign_dict.keys())))

        # Add variables to the execution graph
        with self.session.graph.as_default():
            self._hidden = dict()
            for var in hidden:
                self._hidden[var] = tf.Variable(var.dtype.as_numpy_dtype(assign_dict[var]),
                                                name=var.name.split(':')[0])
        self.session.run(tf.initialize_variables(list(self._hidden.values())))
        # Sort the hidden variables so we can access them in a consistant order
        self._hidden_sorted = sorted(self._hidden.keys(), key=lambda v: v.name)
        for h in self._hidden.values():
            with self.session.graph.as_default():
                var = tf.Variable(h.dtype.as_numpy_dtype(),
                                  name=h.name.split(':')[0] + '_placeholder')
                setter = h.assign(var)
            self._setters[h] = (setter, var)

        all_vars = self._hidden.copy()
        all_vars.update(self._observed)

        self._rewrite_graph(all_vars)

        with self.session.graph.as_default():
            # observed_logps contains one element per data point
            observed_logps = OrderedDict()
            # TODO Remove, see Model.pdf
            observed_logp_setters = []
            for v in self._observed:
                logp_flag = tf.Variable(
                    np.int32(-42),
                    name=v.name.split(':')[0] + '_logp'
                )
                var = tf.Variable(
                    np.int32(-42),
                    name=logp_flag.name.split(':')[0] + '_placeholder'
                )
                setter = logp_flag.assign(var)

                observed_logp_setters.append((setter, var, logp_flag))
                observed_logps[v] = tf.cond(
                    tf.equal(logp_flag, -42),
                    lambda: self._get_rewritten(self._description[v].logp),
                    lambda: tf.Print(tf.fill(tf.reshape(tf.to_int32(logp_flag), [1]), config.dtype(0)),
                                     [tf.fill(tf.reshape(tf.to_int32(logp_flag), [1]), config.dtype(0))])
                )
            # hidden_logps contains a single value
            hidden_logps = [self._get_rewritten(self._description[v].logp) for v in self._hidden]

            # Handle the case where we don't have observed variables.
            # We define the probability to not observe anything as 1.
            if observed_logps:
                observed_logps = list(observed_logps.values())
            else:
                observed_logps = [tf.constant(0, dtype=config.dtype)]
            self._logp_flag_setters = observed_logp_setters

            self._pdf = tf.exp(tf.add_n(observed_logps))

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

        if observed_logp_setters:
            self.session.run(tf.initialize_variables([x[2] for x in observed_logp_setters]))

        self.initialized = True

    def assign(self, assign_dict):
        '''Set the state of specific unobserved (latent) variables to the specified
        values.

        Parameters
        ----------
        assign_dict : dict
            A dictionary from random variables to values.
            This has to specify a value for a subset of the unobserved (latent)
            variables of the model.
        '''
        if Model._current_model == self:
            raise ModelError("Can't call `model.assign()` inside the model block")

        if not isinstance(assign_dict, dict) or not assign_dict:
            raise ValueError("Argument to assign must be a dictionary with "
                             "more than one element")

        if self._observed is None:
            raise ModelError("Can't assign state to the model before "
                             "`model.observed()` has been called.")

        if self._hidden is None:
            raise ModelError("Can't assign state to the model before "
                             "`model.initialize()` has been called.")

        # Assign values without adding to the graph
        setters = [self._setters[self._hidden[k]][0] for k, v in assign_dict.items()]
        feed_dict = {self._setters[self._hidden[k]][1]: v for k, v in assign_dict.items()}
        for s in setters:
            self.session.run(setters, feed_dict=feed_dict)

    @property
    def state(self):
        '''The current state of every unobserved (latent) variable of the
        model. This is a dict from random variables to values.

        Example
        -------
        >>> # Assume we have a random variable X with value 42
        >>> model.state[X]
        42
        '''
        keys = self._hidden.keys()
        variables = list(self._hidden.values())
        values = self.session.run(variables)
        return {k: v for k, v in zip(keys, values)}

    def _check_data(self, data):
        if self._hidden is None:
            raise ModelError("Can't use the model before it has been "
                             "initialized with `model.initialize(...)`")
        # TODO(ibab) make sure that args all have the correct shape
        if len(data) != len(self._observed):
            raise ValueError("Different number of arguments passed to model "
                             "method than declared in `model.observed()`")

    def _set_data(self, data):
        self._check_data(data)
        ops = []
        feed_dict = {self._setters[k][1]: v
                     for k, v in zip(self._observed.values(), data)
                     if v is not None}
        for obs, arg in zip(self._observed.values(), data):
            if arg is None:
                continue
            ops.append(self._setters[obs][0])
        for s in ops:
            self.session.run(s, feed_dict=feed_dict)

    def _run_with_data(self, expr, data):
        self._check_data(data)
        feed_dict = {k: v for k, v in zip(self._observed.values(), data) if v is not None}
        return self.session.run(expr, feed_dict=feed_dict)

    def pdf(self, *args_in):
        '''The probability density function for observing a single entry
        of each random variable that has been declared as observed.

        This allows you to easily plot the probability density function.

        Parameters
        ----------
        args : lists or ndarrays
            The entries for which we want to know the values of the probability
            density function. All arguments must have the same shape.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> xs = np.linspace(-1, 1, 200)
        >>> plt.plot(xs, model.pdf(xs))
        '''
        # If there is a None included in args we can use
        # `self._logp_flag_setters` to disable parts of the likelihood
        # and crudely integrate out dimensions.
        #
        # If the flags are set equal to -42 the term in the likelihood is
        # replaced with a tensor of zeros, where the size of the tensor is
        # equal to the flag. This size is determined using the length of the
        # first non-`None` element of `args`, if all elements are `None` a
        # default value of 1 is used.
        #
        # TODO Remove this horrible hack and have a better way of integrating
        # out dimensions
        setters = []
        feed_dict = {}
        default_size = ([len(a) for a in args_in if a is not None] or [1])[0]
        for arg, (setter, var, lop_setter) in zip(args_in, self._logp_flag_setters):
            setters.append(setter)
            feed_dict[var] = default_size if arg is None else -42

        self.session.run(setters, feed_dict=feed_dict)

        # A value is still needed for unused datasets so replace `None` with
        # -1 in args
        result = self._run_with_data(self._pdf, [-1 if a is None else a for a in args_in])

        # Set all the flags back to -1 to enable all parts of the likelihood
        self.session.run(setters, feed_dict={k: -42 for k in feed_dict})

        return result

    def nll(self, *args):
        '''The negative log-likelihood for all passed datasets.

        Parameters
        ----------
        args : lists or ndarrays
            The datasets for which we want to know the value of the negative
            log-likelihood density function. The arguments don't need to have
            the same shape.
        '''
        return self._run_with_data(self._nll, args)

    def fit(self, *args, **kwargs):
        '''Perform a maximum likelihood or maximum a posteriori estimate
        using one of the available function optimization backends.

        Parameters
        ----------
        args : lists or ndarrays
            The datasets from which we want to infer the values of unobserved
            (latent) variables. The arguments don't need to have the same
            shape.
        use_gradient : bool
            Whether the optimizer should use gradients derived using
            TensorFlow. Some optimizers may not be able to use gradient
            information, in which case this argument is ignored.
        optimizer : subclass of BaseOptimizer
            The optimization backend to use.
            See the `optimizers` module for which optimizers are available.
        '''
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
        '''Perform MCMC sampling of the possible values of unobserved (latent)
        variables using one of the available sampling backends.

        Parameters
        ----------
        args : lists or ndarrays
            The datasets from which we want to infer the values of unobserved
            (latent) variables. The arguments don't need to have the same
            shape.
        sampler : subclass of BaseSampler
            The sampling backend to use.
            See the `samplers` module for which samplers are available.
        '''
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
