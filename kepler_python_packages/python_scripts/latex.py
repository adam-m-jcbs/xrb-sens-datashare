"""
module to provide LaTeX formatting utilities

In particular for use with IPython
"""

import re
from collections.abc import Iterable, Mapping

def xform(f, v):
    """
    do LaTeX formatting

    TODO - add more special cases
    """
    f = f.replace(r':e}', r':.2e}')
    s = f.format(v)
    m,e = s.split('e')
    e = e.replace('-0', '-')
    e = e.replace('+0', '')
    e = e.replace('+', '')
    if e == '1':
        e = '0'
        m = m[0] + m[2] + m[1] + m[3:]
    if e == '-1':
        e = '0'
        m = '0' + m[0] + m[2:]
    if e == '0':
        s = m
    else:
        s = r'{}\times10^{{{}}}'.format(m, e)
    return s

def lform(form, *values, **kvalues):
    """
    format input string to LaTeX for numerical values

    Capture only {:.*e} cells.
    """
    # form = form.replace(r'{}', r'{:}')
    p = r'(\{([^{}:]*):([0-9]*\.?[0-9]*e)\})'
    p = re.compile(p)
    matches = p.findall(form)
    s = form
    for i,m in enumerate(matches):
        if m[1] == '':
            v = values[i]
        elif m[1] in kvalues:
            v = kvalues[m[1]]
        else:
            v = values[int(m[1])]
        f = r'{{:{}}}'.format(m[2])
        r = xform(f, v)
        s = s.replace(m[0], r, 1)
    return(s)
