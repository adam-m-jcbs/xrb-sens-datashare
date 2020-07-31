"""
Generic interface routines for f2py access

(do not depend on KEPLER)
"""

import sys

import numpy as np

from select import select

def z2int(a):
    """
    convert real64 align to int32

    (this is pointlessly wasteful in space)
    """
    s = a.shape
    o = 'F'
    d = np.ndarray(s, buffer=a.data, dtype = np.int32, order = o)
    return d

def z2str(a):
    """
    convert real64 align to byte*8 to str

    (this is pointlessly wasteful in space)
    """
    s = a.shape
    o = 'F'
    d = np.ndarray(s, buffer=a.data, dtype = np.bytes_, order = o)
    return d

def b2s(npbytes):
    """
    Helper function to convert numpy byte array dtype='|S1' to string.
    """
    return npbytes.tobytes().decode().strip()

def s2b(npbytes, s):
    """
    Helper function to store string in numpy byte array dtype='|S1'.
    """
    npbytes[:len(s)] = s
    npbytes[len(s):] = ' ' * (len(npbytes) - len(s))
    return npbytes

def getch():
    i, o, e = select([sys.stdin], [], [], 0)
    if i:
        ch = sys.stdin.read(1)
    else:
        ch = ''
    return ch

def gets():
    s = ''
    i, o, e = select([sys.stdin], [], [], 0)
    if i:
        while True:
            c = sys.stdin.read(1)
            if c == '\n':
                break
            s += c
    return s
