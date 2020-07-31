"""
12 M_sun star for Marco
"""

# kepgen command

kepgen.MakeRun(mass=12,
               dirtarget='~/kepler/pignatari/s12', 
               composition='solar', 
               bgdir = '~/kepler/pignatari',
               genburn = 'x-02GN93g',
               yeburn = True,
               lane_rhoc = 1.e-3,
               special = 'nugrid')

# test explosion

parm = dict(mass = 12,
            dirtarget='~/kepler/pignatari/s12',
            composition='solar',
            run = True)

kepgen.TestExp(exp=['D'], **parm)

kepgen.TestExp(exp=['A'], **parm)
kepgen.TestExp(exp=['B', 'C'], **parm)

parmH = dict(envelalpha=0.3, **parm)
kepgen.TestExp(exp=['J'], **Hparm)
kepgen.TestExp(exp=['E', 'F', 'G', 'H', 'I'], **parmH)

# now we have write a sequence script with dependences...

# real explosions

kepgen.BurnExp(exp=['D'], **parm) 
kepgen.BurnExp(exp=['A','B','C','E','F','G','H','I','J'], **parm) 


parm = dict(mass = 12,
            dirtarget='~/kepler/pignatari/s12',
            composition='solar',
            run_only = True)

kepgen.BurnExp(exp=['A','B','C','E','F','G','H','I','J'], **parm) 

# dump files

s = sekepdump.loaddump('/home/alex/kepler/pignatari/s12/s12#presn')
s.sewrite('/home/alex/kepler/pignatari/s12')

s = sekdata.SekData('/home/alex/kepler/pignatari/s12/s12.sek')
s.sewrite('/home/alex/kepler/pignatari/s12')

s = sekepdump.loaddump('/home/alex/kepler/pignatari/s12/s12D#nucleo')
s.sewrite('/home/alex/kepler/pignatari/s12', comment='12E50erg')

s = sekepdump.loaddump('/home/alex/kepler/pignatari/s12/s12D#final')
s.sewrite('/home/alex/kepler/pignatari/s12', comment='12E50erg')

s = sekdata.SekData('/home/alex/kepler/pignatari/s12/s12D.sek')
s.sewrite('/home/alex/kepler/pignatari/s12', comment = '12e50erg')

exp=['A','B','C','E','F','G','H','I','J']
cmt=[x + 'e50erg' for x in ['3','6','9','15','18','24','50','100']]
for e,c in zip(exp,cmt):
    s = sekepdump.loaddump('/home/alex/kepler/pignatari/s12/s12' + e + '#nucleo')
    s.sewrite('/home/alex/kepler/pignatari/s12', comment = c)
    s = sekepdump.loaddump('/home/alex/kepler/pignatari/s12/s12' + e + '#final')
    s.sewrite('/home/alex/kepler/pignatari/s12', comment = c)
    s = sekdata.SekData('/home/alex/kepler/pignatari/s12/s12' + e + '.sek')
    s.sewrite('/home/alex/kepler/pignatari/s12', comment = c)

