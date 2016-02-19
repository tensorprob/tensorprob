from collections import defaultdict
import inspect


NAME_COUNTERS = defaultdict(lambda: 0)


def generate_name():
    """Generate a unique name for the object in question

    Returns a name of the form "{calling_class_name}_{count}"
    """
    global NAME_COUNTERS
    calling_class = inspect.stack()[1][0].f_locals['self'].__class__.__name__
    NAME_COUNTERS[calling_class] += 1
    return '{0}_{1}'.format(calling_class, NAME_COUNTERS[calling_class])
