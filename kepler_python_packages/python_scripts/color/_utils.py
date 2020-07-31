import numpy as np
from collections import Iterable

# the following now is a singleton metaclass
class MetaSingletonHash(type):
    """
    Singleton metaclass based on hash

    First creates object to be able to test hash.

    If same hash is found, return old object and discard new one,
    otherwise return old one.

    Usage:
       class X(Y, metaclass = MetaSingletonHash)

    class X needs to provide a __hash__ function
    """
    def __call__(*args, **kwargs):
        cls = args[0]
        try:
            cache = cls._cache
        except:
            cache = dict()
            cls._cache = cache
        obj = type.__call__(*args, **kwargs)
        key = (cls.__name__, obj.__hash__())
        return cache.setdefault(key, obj)


class Slice(object):
    """
    Slice iterator object.
    """
    def __init__(self, *args, **kwargs):
        """
        Construct from slice indices or slice object.  Provide
        optional object size.
        """
        if len(args) == 1 and isinstance(args[0], slice):
            self.slice = args[0]
        else:
            self.slice = slice(*args)
        self.size = kwargs.pop('size', None)
        assert len(kwargs) == 0
    def __iter__(self):
        """
        Slice object iterator.
        """
        if self.size is None:
            self.size = max(self.slice.start, self.slice.stop) + 1
        xslice = self.slice.indices(self.size)
        for i in range(*xslice):
            yield i
    def iter(self, size = None):
        """
        Return iterator with defined object size.
        """
        size = self.size
        for i in self.__iter__():
            yield i
        self.size = size


def iterable(x):
    """
    convert things to an iterable, but omit strings

    May need to add other types.
    """
    if isinstance(x, str):
        x = (x,)
    if isinstance(x, np.ndarray) and len(x.shape) == 0:
        x = (x,)
    if not isinstance(x, (Iterable, np.ndarray)):
        x = (x,)
    return x

def is_iterable(x):
    """
    return whether is a true iterable incluidng numpy.ndarra, but not string
    """
    if isinstance(x, np.ndarray) and len(x.shape) == 0:
        return False
    return isinstance(x, (Iterable, np.ndarray)) and not isinstance(x, str)
