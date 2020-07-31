import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import os, os.path
import io

class MesaHistory(object):
    types = {
        'model_number'   : np.int64,
        'num_retries'    : np.int64,
        'num_backups'    : np.int64,
        'version_number' : np.int64,
        }
    def __init__(self, filename = None, mass = None, sentinel = 's'):
        if filename is None:
            if not isinstance(mass, str):
                mass = str(mass)
            filename = '~/mesa/s{}/LOGS/history.data'.format(mass)
        filename = os.path.expanduser(os.path.expandvars(filename))
        with open(filename, 'rt') as f:
            f.readline()
            self.header_fields = f.readline().split()
            header_dtype = np.dtype([(x, self.types.get(x, np.float64))
                                     for x in self.header_fields])
            header = np.loadtxt(
                io.StringIO(f.readline()),
                dtype = header_dtype,
                ).view(np.recarray)
            f.readline()
            f.readline()
            self.fields = f.readline().split()
            data_dtype = np.dtype([(x, self.types.get(x, np.float64))
                                   for x in self.fields])
            data = np.loadtxt(
                f,
                dtype = data_dtype
                ).view(np.recarray)

        for f in header.dtype.names:
            self.__setattr__(f, header[f][()])
        for f in data.dtype.names:
            self.__setattr__(f, data[f])

    def __len__(self):
        return len(self.__getattribute__(self.fields[0]))

    def mass(self):
        if self.initial_mass == int(self.initial_mass):
            return int(self.initial_mass)
        else:
            return self.initial_mass
