"""
Some interface routines that depend on KEPLER
"""

# STDL import
import numpy as np

# from .kepbin import _kepler

class Item():
    def __init__(self, key, value):
        self._value = value
        self.__doc__ = key
    def __get__(self, instance, owner):
        return self._value[()]
    def __set__(self, *args):
        self._value[()] = args[-1]

class CharVar():
    def __init__(self, var):
        self.var = np.ndarray(((var.shape[-1],) + var.shape[:-1]), dtype=var.dtype, buffer=var.data).reshape(var.shape)
        self.shape = var.shape[:-1]
    def __getitem__(self, index):
        return self.var[index].tobytes().decode().strip()
    def __setitem__(self, index, value):
        self.var[index][:len(value)] = value
        self.var[index][len(value):] = ' ' * (len(self.var[index]) - len(value))
    def __str__(self):
        return self.__class__.__name__ + '(' + ')'

class Parameter():
    def __init__(self, _kepler, xset, sentinel = ''):
        self._data = dict()
        self._index = list()
        self._kepler = _kepler
        xmap = {'time' : 'timesec'}
        zmap = {'nsetparm': 'setparm',
                }
        sentinel = sentinel.strip()
        if len(sentinel) > 0:
            sentinel += ' '
        self.sentinel = sentinel
        for i,(k,t) in enumerate(xset.items()):
            x = k.split('.')
            if len(x) == 1:
                s = 'vars'
            else:
                s,k = x
            com = getattr(self._kepler, s)
            if t == -1:
                v = None
            elif t == 0:
                k1 = zmap.get(k, 'z' + k)
                try:
                    data = getattr(com, k1)
                except AttributeError:
                    data = getattr(com, k1[:8])
                v = np.ndarray(
                    data.shape,
                    dtype = np.int32,
                    buffer = data.data)
            else:
                k1 = xmap.get(k, k)
                v = getattr(com, k1)
            self._data[k] = (v, t)
            self._index.append(k)
            if v is not None:
                setattr(self, k, Item(k, v))

    def __getitem__(self, key):
        try:
            i = int(key)
            key = self._index[i]
        except (TypeError, ValueError):
            pass
        v,t = self._data[key]
        if v is None:
            raise KeyError("key '{}' is not defined.".format(key))
        return v[()]

    def __call__(self, *args, remote = False, **kwargs):
        return self.__getitem__(*args, **kwargs)

    def __setitem__(self, key, value):
        try:
            i = int(key)
            key = self._index[i]
        except (TypeError, ValueError):
            pass
        v = self._data[key][0]
        if v is None:
            raise KeyError("key '{}' is not defined.".format(key))
        v[()] = value

    def __getattribute__(self, key):
        "Emulate type_getattro() in Objects/typeobject.c"
        v = super().__getattribute__(key)
        if hasattr(v, '__get__'):
            return v.__get__(v, self)
        return v

    def __setattr__(self, key, value):
        try:
            v = object.__getattribute__(self, key)
            if hasattr(v, '__set__'):
                return v.__set__(v, self, value)
        except:
            pass
        return super().__setattr__(key, value)

    def name(self, index, remote = False):
        try:
            i = int(index)
            key = self._index[i]
        except (TypeError, ValueError):
            raise KeyError(f"index '{index}' not found.")
        return key

    def number(self, key, remote = False):
        try:
            i = self._index.index(key)
        except ValueError:
            raise KeyError(f"key '{key}' not found.")
        return i

    key = name
    index = number

    def print(self, key = None):
        if key is None:
            for i in range(1, len(self._index)):
                self.print(i)
            return
        try:
            i = int(key)
            key = self._index[i]
        except (TypeError, ValueError):
            i = -1
        v,t = self._data[key]
        if i == -1:
            i = self._index.index(key)
        if v is None:
            raise KeyError(f"key/index is not defined: {key}")
        print(f"{key:<8s} ({self.sentinel}{i:>3d}) = ", v)
