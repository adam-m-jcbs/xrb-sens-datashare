"""
make mass table
"""

import os.path
import io
import urllib, urllib.error
import numpy as np
import subprocess

from isotope import ion
from utils import CachedAttribute, iterable, cachedmethod
from abuset import IonList


class MassTable(object):

    def __init__(self, version = 'Ame16', filename = None, path = '~'):
        """
        Make mass table from

        Ame03:
            http://www.nndc.bnl.gov/masses/mass.mas03
            "The Ame2003 atomic mass evaluation (II)"
            by G.Audi, A.H.Wapstra and C.Thibault
            Nuclear Physics A729 p. 337-676, December 22, 2003.

        Ame12:
            http://www.nndc.bnl.gov/masses/mass.mas12
            "The Ame2012 atomic mass evaluation (I)"
            by G.Audi, M.Wang, A.H.Wapstra, F.G.Kondev,
            M.MacCormick, X.Xu, and B.~Pfeiffer
            Chinese Physics C36 p. 1287-1602, December 2012.

        Ame16:
            https://www-nds.iaea.org/amdc/ame2016/mass16.txt
            "The Ame2016 atomic mass evaluation (I)"
            by W.J.Huang, G.Audi, M.Wang, F.G.Kondev, S.Naimi and X.Xu
            Chinese Physics C41 030002, March 2017

        Store values in MeV for total nucleus

        Raw Data:
           Assume file name is ${HOME}/mass.masXX.txt

        Output
           Write out result in ${HOME}/mass.masXX.dat

        TODO - make command line interface
        """

        urls = {
            'Ame95' : 'http://amdc.impcas.ac.cn/masstables/Ame1995/mass_rmd.mas95',
            # 'Ame03' : 'http://www.nndc.bnl.gov/masses/mass.mas03',
            'Ame03' : 'http://amdc.impcas.ac.cn/masstables/Ame2003/mass.mas03',
            # 'Ame12' : 'http://www.nndc.bnl.gov/masses/mass.mas12',
            'Ame12' : 'http://amdc.impcas.ac.cn/masstables/Ame2012/mass.mas12',
            # 'Ame16' : 'http://www-nds.iaea.org/amdc/ame2016/mass16.txt',
            'Ame16' : 'http://amdc.impcas.ac.cn/masstables/Ame2016/mass16.txt',
            }
        url = urls[version]

        if filename is not None:
            with open(filename, 'r') as f:
                content = f.read()
        else:
            filename = os.path.join(os.path.expanduser(path),f'mass.{version}.txt')
            if os.path.isfile(filename):
                print(f'[MassTable] Using {filename}')
                with open(filename, 'rt') as f:
                    content = f.read()
            else:
                print(f'[MassTable] Downloading {url} ... ', end = '', flush = True)
                try:
                    response = urllib.request.urlopen(url)
                    content = response.read().decode()
                    with open(filename, 'wt') as f:
                        f.write(content)
                except urllib.error.HTTPError:
                    args = ['wget', '-O', filename, url]
                    subprocess.check_call(args)
                    with open(filename, 'rt') as f:
                        content = f.read()
                print('done.')
                print(f'[MassTable] Writing to {filename}')
        print('[MassTable] Compiling data ... ', end = '', flush = True)
        data = dict()
        with io.StringIO(content) as f:
            # read header
            for i in range(39):
                f.readline()
            for line in f:
                N = int(line[4:9])
                Z = int(line[9:14])
                A = int(line[14:19])
                # mass excess
                me = float(line[28:41].replace('#','.'))
                # uncertainty
                med = float(line[41:52].replace('#','.'))
                # binding energy/nucleon
                be = float(line[52:63].replace('#','.'))
                # uncertainty
                bed = float(line[63:72].replace('#','.'))
                # amu mass excess
                m = int(line[96:99])
                mum = float(line[100:112].replace('#','.'))
                mumd = float(line[112:].replace('#','.'))
                data[ion(A = A, Z = Z)] = {
                    'mass' :  m + mum * 1.e-6,
                    'mass_excess' : me * 1e-3,
                    'binding_energy' : be * 1.e-3 * A,
                    'mass_uncertainty' : mumd * 1.e-6,
                    'mass_excess_uncertainty' : med * 1.e-3,
                    'binding_energy_uncertainty' : bed * 1.e-3 * A,
                    }
        print('done.')

        self.version = version
        self.data = data
        self.url = url

    def write_be(self, filename = None):
        if filename is None:
            filename = os.path.join(
                os.path.expanduser('~'),
                f'mass.{self.version}.dat')
        data = [(k, v['binding_energy']) for k,v in self.data.items()]
        data.sort()
        with open(filename, 'w') as F:
            F.write(f'; generated from {self.url}\n')
            for d in data:
                F.write('{:<8s} {:17.13f}\n'.format(
                    d[0].Name(), d[1]))

    @CachedAttribute
    def IonList(self):
        """
        Return IonList
        """
        return IonList(self.data.keys())

    ions = IonList

    #@cachedmethod
    def mass_excess(self, ions = None):
        """
        Return mass excess
        """
        if ions is None:
            me = [d['mass_excess'] for d in self.data.values()]
        else:
            ions = ion(np.array(iterable(ions)))
            me = [self.data[i]['mass_excess'] for i in ions]
        return np.array(me)

    @CachedAttribute
    def ME(self):
        """
        Return mass excess
        """
        return self.mass_excess()

    def __getitem__(self, key):
        key = ion(key)
        return self.data[key]
