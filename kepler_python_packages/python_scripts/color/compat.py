#######################################################################

# these are sort of obsolecent

import numpy as np

from matplotlib.colors import rgb2hex

from .models import color_model

def isocolors(n, start=0, stop=360):
    h = np.linspace(start, stop, n, endpoint = False)
    return np.array([rgb2hex(color_model('HSV')(hi,1,1)) for hi in h ])

def isogray(n, start=0, stop=1, endpoint=False):
    h = np.linspace(start, stop, n, endpoint = endpoint)
    return np.array([rgb2hex(g) for g in color_model('RGB').gray(h)])

def isoshadecolor(n, start=0, stop=1, hue = 0, endpoint = False):
    h = np.linspace(start, stop, n, endpoint = endpoint)
    return np.array([rgb2hex(color_model('HSV')(hue,1-hi, 1)) for hi in h[::-1] ])
