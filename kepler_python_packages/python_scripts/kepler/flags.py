"""
Module to keep kepler flags

To change settings set

   kepler.NBURN = nnnn

before importing kepler.code
"""

import os
import os.path
import re

NBURN = None
JMZ = None
FULDAT = None
NAME = None

try:
    kepler_source_path  = os.path.join(
        os.environ['KEPLER_PATH'],
        'source')
    makefile_path = os.path.join(
        kepler_source_path,
        'Makefile.make')
    makefile = open(makefile_path, 'rt').read()
    JMZ = re.findall(
        '^\s*JMZ\s*=\s*(\d+)\s*$',
        makefile,
        re.MULTILINE + re.DOTALL,
        )[0]
    NBURN = re.findall(
        '^\s*NBURN\s*=\s*(\d+)\s*$',
        makefile,
        re.MULTILINE + re.DOTALL,
        )[0]
    FULDAT = re.findall(
        '^\s*FULDAT\s*:=\s*(\S+)\s*$',
        makefile,
        re.MULTILINE + re.DOTALL,
        )[0]
except:
    raise


def setup(**kwargs):
    global NBURN, JMZ, FULDAT, NAME
    NBURN = kwargs.get('NBURN', NBURN)
    JMZ = kwargs.get('JMZ', JMZ)
    FULDAT = kwargs.get('FULDAT', FULDAT)
    NAME = kwargs.get('NAME', FULDAT)
