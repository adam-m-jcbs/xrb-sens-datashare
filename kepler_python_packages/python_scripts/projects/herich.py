import isotope
import abusets
import kepgen
import os.path

base = os.path.expanduser('~/kepler/projects/herich')

Z = 6.e-4
Yvals = (0.248, 0.4)

def makebgs():

    for Y in Yvals:
        x = abusets.ScaledSolarHelium(Z=Z, helium=Y)
        x.write_bg(os.path.join(base, f'a{int(Y*1000):d}bg'),
                   overwrite = True)

def makeruns():

    masses = (12,)
    for Y in Yvals:
        seq = f'a{int(Y*1000):d}'
        bg = seq + 'bg'
        for m in masses:
            kepgen.MakeRun(
                mass = m,
                bgdir = base,
                genburn = bg,
                projectdir = base,
                series = 'a',
                subdir = seq,
                )
