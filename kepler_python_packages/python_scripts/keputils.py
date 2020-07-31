"""
utilities for kepler tools
"""

from utils import iterable

def mass_equal(mass1, mass2):
    """
    Compare mass values within round-off.
    (strings are converted to float)
    """
    if isinstance(mass1, str):
        mass1 = float(mass1)
    if isinstance(mass2, str):
        mass2 = float(mass2)
    return abs(mass1 - mass2) / (mass1 + mass2) < 1.e-12


def mass_string(masses, decimals = None, powers = False):
    """
    convert mass number to string
    """
    masses = iterable(masses)
    if decimals is None:
        decimals = 0
    xmass = []
    for mass in masses:
        if isinstance(mass, str):
            mass = float(mass)
        xm = "{:.6f}".format(mass).rstrip('0').rstrip('.')
        if decimals > 0:
            if xm.count('.') == 0:
                xm += '.' + '0' * decimals
        if powers:
            if xm.count('.') == 0:
                if xm.endswith('0' * 24):
                    xm = xm[:-24] + 'Y'
                elif xm.endswith('0' * 21):
                    xm = xm[:-21] + 'Z'
                elif xm.endswith('0' * 18):
                    xm = xm[:-18] + 'E'
                elif xm.endswith('0' * 15):
                    xm = xm[:-15] + 'P'
                elif xm.endswith('0' * 12):
                    xm = xm[:-12] + 'T'
                elif xm.endswith('0' * 9):
                    xm = xm[:-9] + 'G'
                elif xm.endswith('0' * 6):
                    xm = xm[:-6] + 'M'
                elif xm.endswith('0' * 3):
                    xm = xm[:-3] + 'k'
        xmass += [xm]
    if len(xmass) == 1:
        xmass =  xmass[0]
    return xmass

def mass_formatter(*args, **kwargs):
    """
    function to format mass string for use with ticker
    """
    return mass_string(args[0])

import matplotlib.ticker

class MassFormatter(matplotlib.ticker.FuncFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(mass_formatter)

class MissingModels(Exception):
    """
    Exception raised for KEPLER data files missing models in sequence
    """
    def __init__(self, models, filename):
        self.models = models
        self.filename = filename

    def __str__(self):
        return (
            'Missing models in file {}: '.format(self.filename) +
            ', '.join(['{:d}'.format(x) for x in self.models])
            )

class RecordVersionMismatch(Exception):
    """
    Exception raised for KEPLER data files with different versions.
    """
    def __init__(self, filename, versions = None, models = None):
        self.models = models
        self.versions = versions
        self.filename = filename

    def __str__(self):
        s = 'Record Version Mismatch in file {}: '.format(self.filename)
        if self.versions is not None:
            s += ', '.join(['{:d}'.format(x) for x in self.versions])
        if self.models is not None:
            s += ', '.join(['{:d}'.format(x) for x in self.models])
        s += '.'
        return s

class UnkownVersion(Exception):
    def __init__(self, version = None, record = None):
        self.version = version
        self.record = record


import os
import glob
import kepdump
import shutil

from fortranfile import FortranReader
from human import byte2human

def truncate_bin(filename, model):
    """
    Truncate KEPLER binary file after provide model number
    """
    pos = None
    with FortranReader(filename) as f:
        while True:
            try:
                f.load()
            except:
                pos = None
                break
            n = f.peek_i4(offset=4)
            if n >= model:
                if n > model:
                    f.backspace()
                pos = f.file.tell()
                break
    if pos is not None:
        if os.path.getsize(filename) > pos:
            with open(filename,'r+b') as f:
                f.seek(pos, os.SEEK_SET)
                f.truncate()
            print('[cutbin] truncated {} to model {} ({})'.format(filename, model, byte2human(pos)))

def cutbin(basename, model):
    ext = ('wnd','cnv','ent','lc ','str', 'log','sek','ent','nu ')
    for e in ext:
        filename = basename + '.' + e
        if os.path.isfile(filename):
            truncate_bin(filename, model)

def combine_bin(filename1, filename2, targetfilename = None):
    """
    add data from file2 at file1

    truncate file1 if needed

    raise if gaps would arise
    """
    with FortranReader(filename2) as f2:
        f2.load()
        n2 = f2.peek_i4(offset=4)
    print('[combine_bin] will try to join at model {}'.format(n2))
    pos = None
    n1 = 0
    with FortranReader(filename1) as f1:
        while True:
            try:
                f1.load()
            except:
                pos = None
                break
            np = n1
            n1 = f1.peek_i4(offset=4)
            if n1 >= n2 - 1:
                if n1 >= n2:
                    raise Excption('Gap in models between {} and {}'.format(
                        np, n2))
                pos1 = f1.file.tell()
                break
    if pos1 is None:
        raise Exception('Gap in models between {} and {}'.format(
            n1, n2))
    if os.path.getsize(filename1) > pos1:
        with open(filename1,'r+b') as f:
            f.seek(pos1, os.SEEK_SET)
            f.truncate()
    if targetfilename is not None:
        shutil.copy2(filename1, targetfilename)
        filename1 = targetfilename
    with open(filename1,'ab') as f1, open(filename2,'rb') as f2:
        f1.write(f2.read())
    return n2

def joinbin(basename1, basename2, targetbasename = None):
    """
    join KEPLER bin data files
    """
    ext = ('wnd','cnv','ent','lc ','str', 'log','sek','ent','nu ')
    targetfilename = None
    for e in ext:
        filename1 = basename1 + '.' + e
        filename2 = basename2 + '.' + e
        if targetbasename is not None:
            targetfilename = targetbasename + '.' + e
        if os.path.isfile(filename1):
            n2 = combine_bin(filename1, filename2, targetfilename)
    return n2

def join(basename1, basename2, targetbasename = None):
    """
    join kepler runs

    TODO - check UUID RUN, logfile
    """
    n2 = joinbin(basename1, basename2, targetbasename)
    if targetbasename is None:
        for fn1 in glob.glob(basename1 + '#*'):
            d = kepdump.load(fn1)
            if d.ncyc >= n2:
                print('[joinbin] deleting {}'.format(fn1))
                os.remove(fn1)
    else:
        for fn1 in glob.glob(basename1 + '#*'):
            d = kepdump.load(fn1)
            if d.ncyc < n2:
                fn2 = fn1.replace(basename1, targetbasename, 1)
                print('[joinbin] copying {} --> {}'.format(fn1, fn2))
                shutil.copy2(fn1, fn2)
        basename1 = targetbasename
    for fn2 in glob.glob(basename2 + '#*'):
        d = kepdump.load(fn2)
        if d.ncyc >= n2:
            fn1 = fn2.replace(basename2, basename1, 1)
            print('[joinbin] copying {} --> {}'.format(fn2, fn1))
            shutil.copy2(fn2, fn1)
