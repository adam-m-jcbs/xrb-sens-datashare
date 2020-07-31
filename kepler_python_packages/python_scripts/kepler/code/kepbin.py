"""
Import desired kepler library.

Allow specifying Makefile build parameters:
  JMZ, NBUNZ, FULDAT
and a NAME keyword

TODO: specify 'profiles'?
"""

import sys
import importlib
import os.path

from ._build import _BuildKepler
from ..flags import NBURN, JMZ, FULDAT, NAME

def build(profile = None, return_params = False, **kwargs):
    if profile is not None:
        path = path = os.path.dirname(__file__)
        filename = os.path.join(path, profile + '.pro')
        options = {}
        with open(filename, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                exec(f'options.update(dict({line}))')
        print(f'[load] Profile options: {options}')
        options.update(kwargs)
        kwargs = options

    kwargs.setdefault('NBURN', NBURN)
    kwargs.setdefault('JMZ', JMZ)
    kwargs.setdefault('FULDAT', FULDAT)
    kwargs.setdefault('NAME', NAME)

    _builder = _BuildKepler(**kwargs)
    _builder.run()

    print(f' [{__name__}] Using JMZ = {kwargs["JMZ"]}, NBURN = {kwargs["NBURN"]}.')

    if return_params:
        return kwargs, _builder

def load(profile = None, return_loaded = False, **kwargs):
    """
    Compile and load requested version, return module.
    """

    modules = set(sys.modules)

    kwargs, _builder = build(
        profile = profile,
        return_params = True,
        **kwargs)

    _kepler = importlib.import_module(f'._kepler{_builder.version}', 'kepler.code')

    if _kepler.__name__ in modules:
        loaded = True
        print(f' [{__name__}] module {_kepler.__name__} was already loaded.')
    else:
        loaded = False

    retval = [_kepler]
    if return_loaded:
        retval += [loaded]

    if len(retval) == 1:
        return retval[0]
    return retval


# load kepler module if requested.  This should not be done.
_kepler_loaded = None
def __getattr__(key):
    if key == '_kepler*':
        global _kepler_loaded
        if _kepler_loaded is None:
            _kepler_loaded = load(
                NBURN = NBURN,
                JMZ = JMZ,
                FULDAT = FULDAT,
                NAME = NAME,
                )
            print(f' [{__name__}] Loaded _kepler module globally.')
        _kepler = _kepler_loaded
        return _kepler
    raise AttributeError()
