#! /bin/env python3
"""
Renumber references in the form

[81] "Going supernova.," Heger, A., Nature, 494, 46{47 (2013).
"""

import re
import time
import os, os.path
import sys


if __name__ == "__main__":
    fni = sys.argv[1]
    if len(sys.argv) < 3:
        fno = fni + '-' + time.strftime('%Y%m%d%H%M%S')
    else:
        fno = sys.argv[2]
    p = re.compile('^\[([0-9]+)\] ')
    n = 0
    with open(fni, 'rt') as fi:
        x = fi.read()
    with open(fno, 'wt') as fo:
        for l in x.splitlines(True):
            x = p.findall(l)
            if len(x) > 0:
                n += 1
                l = l.replace(x[0], '{:d}'.format(n))
            fo.write(l)
