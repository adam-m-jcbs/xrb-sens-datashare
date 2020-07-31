"""
Module for weak data.
"""

import numpy as np
import copy

class WeakGrid(object):
    """
    Collect T - rho*Ye(=ne) information on a grid based on dumps and central data
    """
    def __init__(self, res = 0.1, tmin = 6, tmax = 11, dmin=0, dmax=11):
        """
        Initialise parameters for empty grid.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.dmin = dmin
        self.dmax = dmax
        self.res = res

        self.nt = round((tmax - tmin) / res + 0.1 )
        self.nd = round((dmax - dmin) / res + 0.1 )

        self.data = np.zeros((self.nt, self.nd), dtype = np.int64)

    def add_dump(self, dump):
        """
        Add a KEPLER binary dump.
        """
        if isinstance(dump, str):
            import kepdump
            dump = kepdump._load(dump)
        ii = dump.center_slice
        self.add_td(dump.tn[ii], dump.dn[ii] * dump.ye[ii])

    def add_conv(self, conv):
        """
        Add a KEPLER convection data file (central data).
        """
        if isinstance(conv, str):
            import convdata
            conv = convdata._load(conv)
        self.add_td(conv.tc, conv.dc * conv.ye)

    def add_td(self, t, ne, multi = False):
        """
        Add T and d(=n_e) information
        """
        d = ne
        iit = np.int_((np.log10(t) - self.tmin) / self.res)
        iid = np.int_((np.log10(d) - self.dmin) / self.res)
        ii, = np.where((iit >= 0) & (iit < self.nt) & (iid >=0) & (iid < self.nd))
        if not multi:
            self.data[iit[ii],iid[ii]] += 1
        else:
            for i in ii:
                self.data[iit[i],iid[i]] += 1

    def print(self):
        """
        Print grid information as ranges.
        """
        print('temp dmin dmax')
        d0 = np.array([self.dmax, self.dmin])
        for i in range(self.nt):
            ii = np.where(self.data[i] > 0)[0]
            if len(ii) == 0:
                d0 = np.array([self.dmax, self.dmin])
                continue
            d1 = self.dmin + self.res * np.array([ii[0], ii[-1] + 1])
            d = np.array([
                np.minimum(d1[0], d0[0]),
                np.maximum(d1[1], d0[1])
                ])
            print('{:4.1f} {:4.1f} {:4.1f}'.format(
                self.tmin + self.res * i, *d))
            d0 = d1

    def __add__(self, other):
        """
        Add two objects.  Need to have same table layout.
        """
        assert isinstance(other, self.__class__)
        assert self.res == other.res
        assert self.tmin == other.tmin
        assert self.dmin == other.dmin
        assert self.data.shape == other.data.shape

        x = copy.deepcopy(self)
        x.data += other.data
        return x

    def plot(self):
        """
        Plot grid information.

        Shading indicates number of grid points
        (beautiful but irrelevant).

        Red outline give grid extent of non-zero values.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = self.dmin + self.res * np.arange(self.nd + 1)
        y = self.tmin + self.res * np.arange(self.nt + 1)
        ax.pcolorfast(x, y, -self.data, cmap='gray')
        ax.set_xlabel('log electron density ($\mathsf{N_{\!A}\,cm^{-3}}$)')
        ax.set_ylabel('log temperature (K)')

        # plot outline
        invalid = None
        d1 = invalid
        t1 = self.tmin
        lp = dict(color = 'r', lw = 1)
        x = [[],[]]
        y = [[],[]]
        for i in range(self.nt):
            d0 = d1
            t0 = t1
            ii, = np.where(self.data[i] > 0)
            t1 = self.tmin + self.res * (i + 1)
            if len(ii) == 0:
                if not d0 is invalid:
                    x[0] += [d0[0], d0[1]]
                    y[0] += [t0] * 2
                d1 = invalid
                continue
            d1 = self.dmin + self.res * np.array([ii[0], ii[-1] + 1])
            if d0 is invalid:
                x[0] += [d1[1]]
                y[0] += [t0]
            x[0] += [d1[0], d1[0]]
            y[0] += [   t0, t1   ]
            x[1] += [d1[1], d1[1]]
            y[1] += [   t0,    t1]

        ax.plot(x[0]+x[1][::-1], y[0]+y[1][::-1], **lp)

        plt.show()

    def add_dir(self, path, dump = True, conv = False, recursive = False):
        """
        Add all files of specified type in directory.
        """
        import glob
        import os.path
        path = os.path.expanduser(os.path.expandvars(path))
        if recursive:
            path = os.path.join(path, '**')
        if dump:
            for p in glob.iglob(os.path.join(path, '*#*'), recursive = recursive):
                print(' [add_dir] Adding {}'.format(p))
                try:
                    self.add_dump(p)
                except:
                    print(' [add_dir] ERROR adding {}'.format(p))
        if conv:
            for p in glob.iglob(os.path.join(path, '*.cnv*'), recursive = recursive):
                print(' [add_dir] Adding {}'.format(p))
                try:
                    self.add_conv(p)
                except:
                    print('[add_dir] ERROR adding {}'.format(p))

    def save(self, filename = 'weak.pk'):
        """
        Write this object to file.
        """
        import os.path
        import pickle
        with open(os.path.expanduser(os.path.expandvars(filename)), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename = 'weak.pk'):
        """
        Load object from file and return it as new object.

        This is a class method and does not change data of an existing
        object or load the data into an exisiting object.
        """
        import os.path
        import pickle
        with open(os.path.expanduser(os.path.expandvars(filename)), 'rb') as f:
            data = pickle.load(f)
        assert isinstance(data, cls), 'ERROR: not a {} data file'.format(cls.__name__)
        return data
