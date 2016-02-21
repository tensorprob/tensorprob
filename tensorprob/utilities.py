from collections import defaultdict


NAME_COUNTERS = defaultdict(lambda: 0)


def generate_name(func=None):
    """Generate a unique name for the object in question

    Returns a name of the form "{calling_class_name}_{count}"
    """
    global NAME_COUNTERS

    calling_name = func.__name__

    NAME_COUNTERS[calling_name] += 1
    return '{0}_{1}'.format(calling_name, NAME_COUNTERS[calling_name])


class classproperty(object):
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)
