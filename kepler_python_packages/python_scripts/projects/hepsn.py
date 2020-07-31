import numpy as np
import kepdata
import os.path

def mkdump():
    sdir = '/travel1/alex/kepler/he2sn/hexnuc'
    odir = '/home/alex/Downloads/dumps'
    runs = [
        'hex+10',
        'hex+07',
        'hex+05',
        'hex+03',
        'hex+00',
        'hex-03',
        'hex-06',
        'hex-10',
        'hex-15',
        'hex-20',
        'hex-25',
        'hex-30',
        'hex-35',
        'hex-40',
        'hex-45',
        'hex-50',
        'hex-55',
        'hex-60',
        'hex-65',
        'hex-70',
        'hex-75',
        'hex-80',
        'hex000',
        ]
    for run in runs:
        filename = os.path.join(sdir, run, 'hex100/hex100#nucleo')
        outfilename = os.path.join(odir, run + '-' + 'hex100@@nucleo')
        args = dict(
            filename = filename,
            outfile = outfilename,
            burn = True,
            )
        kepdata.kepdata(**args)
