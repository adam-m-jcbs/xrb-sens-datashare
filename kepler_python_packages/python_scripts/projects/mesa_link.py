m = keplink.ProfileMesa('/home/alex/kepler/nugrid/M15.0Z0.02FULL03.data')
m.dezone(2.e30,1,1.2)
m.dezone(5.e29,1,2)
m.dezone(1.e30,2,4)
m.dezone(1.e29,0,.1)
m.write('/home/alex/kepler/nugrid/s15/s15.dat')

m = keplink.ProfileMesa('/home/alex/kepler/nugrid/M20.0Z0.02FULL03.data')
m.dezone(1.e29,0,.1)
m.dezone(2.e30,1,1.3)
m.dezone(1.e30,1,2)
m.dezone(1.5e30,2,2.5)
m.dezone(2.e30,4,6)
m.write('/home/alex/kepler/nugrid/s20/s20.dat')

m = keplink.ProfileMesa('/home/alex/kepler/nugrid/M25.0Z0.02FULL03.data')
m.dezone(1.e29,0,.1)
m.dezone(2.e30,1,1.2)
m.dezone(1.e30,1.3,1.4)
m.dezone(5.e29,1,2)
m.dezone(5.e30,2.5,3)
m.dezone(3.e30,6,7.5)
m.dezone(2.e28,8,9)
m.write('/home/alex/kepler/nugrid/s25/s25.dat')

# NOTE: set p 222 (totm0) to appropriate mass (g) in link file

parm = dict(mass = 15,
            dirtarget='~/kepler/nugrid/s15',
            composition='solar',
            run = True)

kepgen.TestExp(exp=['D'], **parm)


parm = dict(mass = 20,
            dirtarget='~/kepler/nugrid/s20',
            composition='solar',
            run = True)

kepgen.TestExp(exp=['D'], **parm)


parm = dict(mass = 25,
            dirtarget='~/kepler/nugrid/s25',
            composition='solar',
            run = True)

kepgen.TestExp(exp=['D'], **parm)

# ========================================

parm = dict(mass = [15, 20, 25],
            series = 's',
            dirbase='~/kepler/nugrid',
            run = True)

kepgen.BurnExp(exp=['D'], **parm)

s = sekdata.SekData('~/kepler/nugrid/s15/s15D.sek')
s.sewrite(comment = 'ExpD', metallicity = 0.02, mass = 15)

s = sekdata.SekData('~/kepler/nugrid/s20/s20D.sek')
s.sewrite(comment = 'ExpD', metallicity = 0.02, mass = 20)

s = sekdata.SekData('~/kepler/nugrid/s25/s25D.sek')
s.sewrite(comment = 'ExpD', metallicity = 0.02, mass = 25)


# ===========
# ===========
# ===========

def tsy(filename):
    import convdata
    import os.path
    """
    Write out time till core collapse, central entropy, and central
    Ye from cnv file

    EXAMPLE
    tsy('/m/kepler/solgn93/snuc/s25/s25.cnv.gz')

    """
    c = convdata.loadconv(filename)
    t = c.timecc()
    s = c.sc
    y = c.ye

    dirname = os.path.dirname
    basename = os.path.basename(filename).split(os.path.extsep)[0]
    outfile = os.path.join(dirname, basename + '_tsy.txt')

    with open(outfile, 'w') as f:
        for x in zip(t,s,y):
            f.write('{:20f} {:20f} {:20f}'.format(*x) + '\n')
