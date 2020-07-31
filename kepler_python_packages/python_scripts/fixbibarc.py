#! /bin/env python3

import re
import sys

format = r'{}'

replacements = {
    r'\\aj' : 'The Astronomical Journal',
    r'\\actaa' : 'Acta Astronomica',
    r'\\araa' : 'Annual Review of Astronomy and Astrophys',
    r'\\apj' : 'The Astrophysical Journal',
    r'\\apjl' : 'The Astrophysical Journal, Letters',
    r'\\apjs' : 'The Astrophysical Journal, Supplement',
    r'\\ao' : 'Applied Optics',
    r'\\apss' : 'Astrophysics and Space Science',
    r'\\aap' : 'Astronomy \\& Astrophysics',
    r'\\aapr' : 'Astronomy \\& Astrophysics Reviews',
    r'\\aaps' : 'Astronomy \\& Astrophysics, Supplement',
    r'\\azh' : 'Astronomicheskii Zhurnal',
    r'\\baas' : 'Bulletin of the AAS',
    r'\\bac' : 'Bulletin of the Astronomical Institutes of Czechoslovakia',
    r'\\caa' : 'Chinese Astronomy and Astrophysics',
    r'\\cjaa' : 'Chinese Journal of Astronomy and Astrophysics',
    r'\\icarus' : 'Icarus',
    r'\\jcap' : 'Journal of Cosmology and Astroparticle Physics',
    r'\\jrasc' : 'Journal of the RAS of Canada',
    r'\\memras' : 'Memoirs of the RAS',
    r'\\mnras' : 'Monthly Notices of the RAS',
    r'\\na' : 'New Astronomy',
    r'\\nar' : 'New Astronomy Review',
    r'\\pra' : 'Physical Review A: General Physics',
    r'\\prb' : 'Physical Review B: Solid State',
    r'\\prc' : 'Physical Review C',
    r'\\prd' : 'Physical Review D',
    r'\\pre' : 'Physical Review E',
    r'\\prl' : 'Physical Review Letters',
    r'\\pasa' : 'Publications of the Astron. Soc. of Australia',
    r'\\pasp' : 'Publications of the ASP',
    r'\\pasj' : 'Publications of the ASJ',
    r'\\rmxaa' : 'Revista Mexicana de Astronomia y Astrofisica',
    r'\\qjras' : 'Quarterly Journal of the RAS',
    r'\\skytel' : 'Sky and Telescope',
    r'\\solphys' : 'Solar Physics',
    r'\\sovast' : 'Soviet Astronomy',
    r'\\ssr' : 'Space Science Reviews',
    r'\\zap' : 'Zeitschrift fuer Astrophysik',
    r'\\nat' : 'Nature',
    r'\\iaucirc' : 'IAU Cirulars',
    r'\\aplett' : 'Astrophysics Letters',
    r'\\apspr' : 'Astrophysics Space Physics Research',
    r'\\bain' : 'Bulletin Astronomical Institute of the Netherlands',
    r'\\fcp' : 'Fundamental Cosmic Physics',
    r'\\gca' : 'Geochimica Cosmochimica Acta',
    r'\\grl' : 'Geophysics Research Letters',
    r'\\jcp' : 'Journal of Chemical Physics',
    r'\\jgr' : 'Journal of Geophysics Research',
    r'\\jqsrt' : 'Journal of Quantitiative Spectroscopy and Radiative Transfer',
    r'\\memsai' : 'Mem. Societa Astronomica Italiana',
    r'\\nphysa' : 'Nuclear Physics A',
    r'\\physrep' : 'Physics Reports',
    r'\\physscr' : 'Physica Scripta',
    r'\\planss' : 'Planetary Space Science',
    r'\\procspie' : 'Proceedings of the SPIE',
    }

def fix_f___ing_arc(filename, outname = None):
    with open(filename, 'rt') as f:
        lines = f.read()
    for k,r in replacements.items():
        r = format.format(r)
        r = f'{{{r}}}'
        k = f'{{{k}}}'
        lines = re.sub(k, r, lines)
    if outname is None:
        outname = filename
    with open(outname, 'wt') as f:
        f.write(lines)

if __name__ == "__main__":
    fix_f___ing_arc(*sys.argv[1:3])
