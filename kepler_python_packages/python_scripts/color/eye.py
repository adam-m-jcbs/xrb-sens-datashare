# http://www.cie.org.au/archive.html

import os.path
import urllib.request
import io
import sys
import numpy as np

datapath = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'data')

# http://www.cvrl.org/
data_sources = {
    'ciexyz31' : {
        'file' : 'ciexyz31.csv',
        'url' : 'http://www.cvrl.org/database/data/cmfs/ciexyz31.csv',
        'comment' : 'ciexyz31 color matching functions 2deg 5nm',
        },
    'ciexyz31_1' : {
        'file' : 'ciexyz31_1.csv',
        'url' : 'http://www.cvrl.org/database/data/cmfs/ciexyz31_1.csv',
        'comment' : 'ciexyz31 color matching functions 2deg 1nm',
        },
    'ciexyz64' : {
        'file' : 'ciexyz64.csv',
        'url' : 'http://www.cvrl.org/database/data/cmfs/ciexyz64.csv',
        'comment' : 'ciexyz64 color matching functions 10deg 5nm',
        },
    'ciexyz64_1' : {
        'file' : 'ciexyz64_1.csv',
        'url' : 'http://www.cvrl.org/database/data/cmfs/ciexyz64_1.csv',
        'comment' : 'ciexyz64 color matching functions 10deg 1nm',
        },
    'Illuminantd65' : {
        'file' : 'Illuminantd65.csv',
        'url' : 'http://www.cvrl.org/database/data/cie/Illuminantd65.csv',
        'comment' : '6504 K, 5nm',
        },
    'Illuminanta' : {
        'file' : 'Illuminanta.csv',
        'url' : 'http://www.cvrl.org/database/data/cie/Illuminanta.csv',
        'comment' : '5nm',
        },
    'scvle' : {
        'file' : 'scvle.csv',
        'url' : 'http://www.cvrl.org/database/data/lum/scvle.csv',
        'comment' : 'CIE (1951) Scotopic V\'(lambda), 5nm',
        },
    'scvle_1' : {
        'file' : 'scvle_1.csv',
        'url' : 'http://www.cvrl.org/database/data/lum/scvle_1.csv',
        'comment' : 'CIE (1951) Scotopic V\'(lambda), 1nm',
        },
    'vl1924e' : {
        'file' : 'vl1924e.csv',
        'url' : 'http://www.cvrl.org/database/data/lum/vl1924e.csv',
        'comment' : 'CIE (1924) Photopic V(lambda), 5nm',
        },
    'vl1924e_1' : {
        'file' : 'vl1924e_1.csv',
        'url' : 'http://www.cvrl.org/database/data/lum/vl1924e_1.csv',
        'comment' : 'CIE (1924) Photopic V(lambda), 1nm',
        },

    }

class Eyedata():
    def __init__(self):
        pass

    def __getattr__(self, name):
        if name in data_sources:
            attr = '_' + name
            try:
                return getattr(self, attr)
            except:
                pass
            sources = data_sources[name]
            filename = os.path.join(datapath, sources['file'])
            delim = sources.get('delim', ',')
            try:
                data = np.loadtxt(filename, delimiter = delim)
                setattr(self, attr, data)
                return data
            except:
                pass
            url = sources['url']
            try:
                response = urllib.request.urlopen(url)
                html = response.read().decode()
                data = np.loadtxt(io.StringIO(html), delimiter = delim)
                setattr(self, attr, data)
                with open(filename, 'wt') as f:
                    f.write(html)
                return data
            except:
                # good for diagnostic
                raise
        raise AttributeError()
    def load_all(self):
        for x in data_sources:
            getattr(self, x)


# http://rspb.royalsocietypublishing.org/content/royprsb/220/1218/115.full.pdf
_rods_cones_mormalized = [
    [370, 37.0, 59.3, -1.0, -1.0],
    [380, 37.0, 65.2, -1.0, -1.0],
    [390, 37.0, 74.4, -1.0, -1.0],
    [400, 38.4, 87.8, 39.7, 43.4],
    [410, 40.1, 97.2, 38.7, 41.9],
    [420, 42.1, 99.8, 38.2, 40.2],
    [430, 45.7, 95.7, 37.9, 38.3],
    [440, 51.5, 85.4, 39.6, 37.7],
    [450, 60.5, 69.5, 42.9, 36.3],
    [460, 71.8, 50.5, 48.1, 38.0],
    [470, 83.0, 35.5, 55.5, 41.0],
    [480, 92.4, 24.8, 63.7, 46.1],
    [490, 98.2, 15.9, 73.1, 52.1],
    [500, 99.5, 10.6, 82.3, 59.4],
    [510, 95.0, 5.3, 90.4, 67.0],
    [520, 85.5, 3.0, 96.9, 75.7],
    [530, 72.5, -1.0, 99.9, 84.4],
    [540, 58.1, -1.0, 98.6, 92.2],
    [550, 43.4, -1.0, 92.4, 97.9],
    [560, 31.5, -1.0, 82.6, 99.9],
    [570, 22.7, -1.0, 70.2, 97.7],
    [580, 15.8, -1.0, 56.0, 91.4],
    [590, 11.0, -1.0, 42.7, 82.2],
    [600, 7.3, -1.0, 31.7, 70.9],
    [610, 4.7, -1.0, 23.2, 58.9],
    [620, 3.7, -1.0, 15.5, 44.9],
    [630, -1.0, -1.0, 11.2, 31.8],
    [640, -1.0, -1.0, 7.5, 21.4],
    [650, -1.0, -1.0, -1.0, 12.7],
    ]
rods_cones_mormalized = np.array(_rods_cones_mormalized)
rods_cones_mormalized[rods_cones_mormalized < 0] = np.nan


# more rod stuff
# https://midimagic.sgc-hosting.com/huvision.htm
# http://www.pnas.org/content/96/2/487
# http://www.pnas.org/content/96/2/487

# transforms
# http://poynton.ca/Poynton-color.html
# http://poynton.ca/PDFs/Guided_tour.pdf

# http://www.cvrl.org/neur3045/Lecture%20Notes/Stockman/Achromatic%20and%20chromatic%20vision.pdf


# http://www.cvrl.org/cie.htm


# luminosity functions
# https://web.archive.org/web/20081228115119/http://www.cvrl.org/database/text/lum/scvl.htm
# http://www.cvrl.org/database/data/lum/scvle_1.txt
