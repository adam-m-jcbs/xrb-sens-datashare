"""
KEPLER python package
"""

# import os
import numpy as np

# # this one does not work - code will just stop on starting kepler
# # kepler start maybe needs a check...
# os.environ['GFORTRAN_CONVERT_UNIT'] = 'big_endian'

np.seterr(over='ignore', invalid='ignore')

from .flags import NBURN, JMZ, FULDAT
