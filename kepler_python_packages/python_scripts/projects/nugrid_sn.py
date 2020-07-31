import kepgen

# kepgen.MakeRun(mass=[12,15,18,20,25], composition='solgn93', special={'nugrid'}, run = True, yeburn=True)

# kepgen.TestExp(mass=[12,15,18,20,25], exp='D', composition='solgn93', special={'nugrid'}, run = True, yeburn=True)

# kepgen.BurnExp(mass=[12,15,18,20,25], exp='D', composition='solgn93', special={'nugrid'}, run = True, yeburn=True)


# v3a
# sshfs c:/g/alex/kepler/solgn93/ solgn93
import os
import os.path
from sekepdump import SeKepDump
path = '/home/alex/kepler/solgn93/snuc'
outpath = '/m/web/Download/NuGrid/v3a/se-models'
dumps = ['0','presn']
masses = ['15','20','25']
files = [os.path.join(path, 's{}r'.format(m), 's{}#{}'.format(m, d)) for m in masses for d in dumps]
for f in files:
    d = SeKepDump(f)
    d.sewrite(path = outpath, burn = True)


from sekdata import SekData
outpath = '/m/web/Download/NuGrid/v3a/se'
files = [os.path.join(path, 's{}r'.format(m), 's{}.sek'.format(m)) for m in masses]
for f in files:
    s = SekData(f)
    s.sewrite(path = outpath)

# XRB
path = '/home/alex/kepler/XRB4/xrba5m/xrba5'
