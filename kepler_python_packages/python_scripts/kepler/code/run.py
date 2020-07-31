#! /bin/env python3

import sys
import importlib
import os
import os.path
import importlib, importlib.util

from kepler.code.main import Kepler
from kepler.cmdloop import KeplerCmdLoop

def load_module(name, path = None):
    if path is None:
        path = os.getcwd()
    path = os.path.expanduser(os.path.expandvars(path))
    filename = os.path.join(path, name + '.py')
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_module2(name, path = None):
    path = os.path.expanduser(os.path.expandvars(path))
    if not path in sys.path:
        sys.path.insert(0, path)
    else:
        path = None
    try:
        module = importlib.import_module(name)
    except:
        raise
    finally:
        if path is not None:
            sys.path.remove(path)
    return module

class KeplerShell(KeplerCmdLoop):
    '''
    support config files, default is config.py

    args = ['s']
    kwargs = dict(
        debug = False,
        )
    def config(k):
        print(f'I wss called for {k}')
        # set hooks here
        # set kepler hook parameters
    '''

    def __init__(self, *args, **kwargs):
        # TODO - maybe load <name>.py for config info
        # unfortunately, <name> itself to could be changed by config file
        config = kwargs.pop('config', None)
        if config is not None:
            require_config = True
        else:
            require_config = False
            config = 'config'
        path = kwargs.get('path', os.getcwd())
        try:
            _config = load_module(config, path)
            print(f' [run] unsing config file {_config.__file__}')
        except (ModuleNotFoundError, FileNotFoundError, ):
            _config = None
        if require_config and _config is None:
            raise AttributeError(f'Could not find config file {config}')
        if _config is not None:
            if hasattr(_config, 'args0'):
                cfg_args = _config.args0
                if isinstance(cfg_args, (list, tuple)):
                    args = tuple(cfg_args) + args
                else:
                    raise AttributeError(f'Invalid args0 in config file {_config.__file__}')
            if hasattr(_config, 'args'):
                cfg_args = _config.args
                if isinstance(cfg_args, (list, tuple)):
                    args = args + tuple(cfg_args)
                else:
                    raise AttributeError(f'Invalid args in config file {_config.__file__}')
            if hasattr(_config, 'kwargs'):
                cfg_kwargs = _config.kwargs
                if isinstance(cfg_kwargs, (dict,)):
                    _kw = type(kwargs)()
                    _kw.update(cfg_kwargs)
                    _kw.update(kwargs)
                    kwargs = _kw
                else:
                    raise AttributeError(f'Invalid kwargs in config file {_config.__file__}')
        self.debug = kwargs.setdefault('debug', True)
        stop = 's' in args
        if stop:
            list(args).remove('s')
        if self.debug:
            print(f' [KeplerShell] {args} {kwargs}')
        kwargs.setdefault('set_defaults', False)
        k = Kepler(*args, **kwargs)
        if k.status == 'started':
            super().__init__(k)
        if _config is not None:
            cfg_func = getattr(_config, 'config', None)
            if cfg_func is not None:
                cfg_func(k)
        if not stop:
            k.g()

def argv2arg(argv):
    args = argv[1:]
    kwargs = dict()
    d = dict()
    for i,a in enumerate(args):
        try:
            args[i] = eval(a, d, d)
        except:
            pass
    for a in list(args):
        try:
            d = eval('dict('+a+')', d, d)
            kwargs.update(d)
            args.remove(a)
            continue
        except:
            pass
        try:
            i = a.index('=')
            s = a[:i+1] + "'" + a[i+1:] + "'"
            d = eval('dict('+s+')', d, d)
            kwargs.update(d)
            args.remove(a)
            continue
        except:
            pass
    kwargs['progname'] = argv[0]
    return args, kwargs

def run(argv):
    args, kwargs = argv2arg(argv)
    KeplerShell(*args, **kwargs).cmdloop()

if __name__ == "__main__":
    # TODO - add argparse
    run(sys.argv)
