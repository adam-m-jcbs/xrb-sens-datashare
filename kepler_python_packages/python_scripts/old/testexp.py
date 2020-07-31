#! /usr/bin/python3
"""
Library of utilities to create test explosion
"""

import logging
import utils
import kepdump
from logged import Logged

class MakeLink(Logged):

    def __init__(
        dumpfile,
        expdir,
        addcutzones = 0,
        zmcut = 0,
        scut = 0,
        mix = 0,
        accel = 0.25,
        surfcut = 0,
        norelativistic = 1,
        envelalpha = 1,
        silent = False
        ):
        # composition
        # mass

        """
        Make explosion setup files and directories.
        """
        self.setup_logger(silent)

        dump = kepdump.loaddump(dumpfile)
        if zmcut > -1:
            icut, = np.where(dump.zm_sun > zmcut)
            if len(icut) <= 0:
                raise AttributeError("zmcut out of bounds")
            icut = icut[0] + addcutzones
            mcut = dump.zm_sun[icut]
            self.logger.info('mass cut at m = {:f} (j={:d})'.format(mcut,icut))
        elif scut == 0:
            icut = dump.core()['ye core'].j-1+addcutzones
            mcut = dump.zm_sun[icut]
            self.logger.info('mass cut at ye core: m = {:f} (j={:d})'.format(mcut,icut))
        else:
            icut, = np.where(dump.stot > scut)
            if len(icut) <= 0:
                raise AttributeError("scut out of bounds")
            icut = icut[0] + addcutzones
            mcut = dump.zm_sun[icut]
            self.logger.info('mass cut at S = {:f}: m = {:f} (j={:d})'.format(scut,mcut,icut))
            if dump.stot[icut]/dump.stot[icut-1] < 1.1:
                self.logger.warning('small entropy jump: S= {:f}, {:f}, {:f}!!!'
                                    .format(dump.stot[icut-1:icut+2]))
        linkfilepath = os.path.join(expdir, 'explosion.link')
        with open(linkfilepath, 'w') as linkfile:
            linkfile.write("""
c link for explosion with alpha = xxx
c
c determine piston and cut inner part
c bounce <j_cut/ye_cut> <t_min> <r_min> <r_max> <alpha>
""")
            if accel == 0:
                sacc = '0.45'
            else:
                sacc = '{:f6.3}'.format(accel)
            linkfile.write('bounce {:d} {:s} 5.d7 1.d9 xxx cut'.format(icut))
            linkfile.write("""
c
c resize graphics window
""")
            linkfile.write('mlim {8.3f} 5.'.format(mcut))
            linkfile.write("""
p 113 12300
p 191 1.5d9
c
c set time to execute the tshock command (at piston bounce) defined below
c (p 343 now set by the bounce command)
c
c turn off hydrostatic equilibrium for the envelope
p 386 -1.e+99

c no opal opacities
c p 377 0

c limit burn co-processing maximum temperature
p 234 1.D10
c
c turn off rotationally induced mixing
p 364 0
c
c turn off convection plot
p 376 0

c set time to execute the tnucleo command(post exp.nucleo)defined below
p 344 100.

c set time to execute the tenvel command(shock in envelop)defined below
p 345 2.5e+4

c change any remaining ise or nse zones to approx
approx 1 99999
c
c zero problem time (add old time to toffset)
zerotime
c
c turn off coulomb corrections and reset zonal energies
p 65 1.e+99
p 215 0.
newe
p 65 1.e+7
c
c dump only every 1000th model
p 156 100
c
c reset other parameter values as required:
c
c reset time-step and back-up controls
p 1 1.e-4
p 6 .05
p 7 .03
p 8  10.
p 25 .002
c
c make less frequent dumps
p 18 1000
p 156 1
p 16 100000000
c
c reset problem termination criteria
p 15 3.e+7
p 306 1.e+99
c
c turn off linear artificial viscosity
p 13 0.
c
c turn off iben opacities
p 29 0.
p 30 0.
c
c set opacity floor
p 50 1.e-5
c
c turn off any boundary temperature or pressure
p 68 0.
p 69 0.
c
c turn off rezoning
p 86 0
c
c turn off convection
p 146 0.
p 147 0.
c
c turn off transition to ise
p 184 1.e+99
c
c reset burn-coprocessing parameters
p 229 10
p 230 .1
p 272 .001
c
c make sure sparse matrix inverter is turned on
p 258 1
c
c set timescale for neutrino pulse used by burn for the nu-process
p 286 3.
c
c set temperature of mu and tau neutrinos for the nu-process
p 288 6.
c
c rest toffset and reference time to zero and use linear time in timeplots
p 315 0.
p 319 0.
p 327 1
c
c reset all dump file names and delete all old dump variables
newdumps
c
c list of dump  variables (name; dump  ratio; dezone ratio; adzone ratio):
c
c thermodynamic quantities
c
c
c Definitions of aliased commands (72 characters max)...
c The tshock command is executed when time reaches tshock (p343)
c The tnucleo command is executed when time reaches tnucleo (p344)
c The tenvel command is executed when time reaches tenvel (p345)
c The tfinal command is executed when time reaches tstop (p15)

c                *********1*********2*********3*********4*********5*********6*********7**
alias  t1       "tq,1,1 i"
alias  tmv      "mlim, ylim, tlim, p 327 3, p 301 40, tm un"
alias  tmd      "mlim, ylim, tlim, p 327 3, p 301 40, tm dn"
alias  tmt      "mlim, ylim, tlim, p 327 3, p 301 40, tm tn"
c
alias  tshock   "p 25 1.e+99, p 229 20"
c
alias  tnucleo  "p 229 100,p 38 0.,p 286 0.,p 64 50,mixnuc,newe"
c
c                         turn off OPAL95
""")
            linkfile.write('alias  tenvel   "p 337 1, p 377 0, mixenv, newe,p 375 {:f}"'.format(envelalpha))
            linkfile.write("""
c
alias  tfinal   "editiso"
c
p 375 .33
p 5 40
p 425 0.
c
c output wind file to make light curve plots
p 390 1
c
c switch off ye taken from BURN -- bad for fallback
p 357 0
c
c allow many steps
p 14 1000000000
c
c long lc output
p 437 50
c
c 5 MeV nu_e_bar
p 446 5.
c
c turn off mass loss
p 363 0.
p 387 0.
p 519 0
c
c less restrictive abundance check for explosions
p 442 1.e-10
""")
            mixnucdum = 'alias mixnuc "p 1"'
            mixenvdum = 'alias mixenv "p 1"'
            mixnuc = 'c'
            mixenv = 'c'
            if mix != 0:
                mixit = 'mix 1 {:d} {:f}'.format(
                    dump.jm-icut,
                    dump.core['He core'].zm_sun * 0.1)
# terminate if mixing fails
                mixit += ' 1'
                if mod(mix,2) == 1:
                    mixenvdum = 'c'
                    mixenv = 'alias mixnuc "mixit, mixit, mixit, mixit"'
                    self.logger.info('applying "'+mixit+'" 4 times at #tenvel.')
                if mod(mix // 2, 3) == 1:
                    mixnucdum = 'c'
                    mixnuc = 'alias mixnuc "mixit, mixit, mixit, mixit"'
                    self.logger.info('applying "'+mixit+'" 4 times at #nucleo.')

            linkfile.write("""
c
c alias for no mixing
""")
            if mixnucdum != 'c':
                linkfile.write(mixnucdum)
            if mixenvdum != 'c':
                linkfile.write(mixenvdum)
            linkfile.write("""
c
c alias for mixing
""")
            if mixnuc != 'c':
                linkfile.write(mixnuc)
            if mixenv != 'c':
                linkfile.write(mixenv)
            linkfile.write('alias mixit "'+mixit+'"')

            if surfcut > 0:
                linkfile.write("""
c
c removing "detached" surface zones...
cutsurf {:d}
p 68 0.
p 69 0.
""".format(surfcut))
                self.logger.warning('REMOVING OUTER {:d} ZONES'.format(surfcut))
            else:
                # cut away surface density jumps ("detached zones")
                i = 0
                j = dump.jm
                while ((dump.dn[j] < 1.e-4) and
                       (dump.ym_sun[j] < 0.1)):
                    if dump.dn[j-1] > 2 * dump.dn[j]:
                        i = dump.jm - j
                        j -= 1
                if i > 0:
                    linkfile.write("""
c
c removing "detached" surface zones...
cutsurf {:d}
p 68 0.
p 69 0.
""".format(i))
            self.logger.warning('REMOVING OUTER {:d} ZONES'.format(i))

            # the following is in particular to fix pulsational pair-SN runs
            # in which surface zones may have been cut away for previous pulses

            if norelativistic == 0:
                linkfile.write("""
c
c do not cut off surface zones
p 271 1.d99
p 409 1.d99
""")
            else:
                linkfile.write("""
c
c do not go relativistic
p vloss 1.d10
""")
        # done writing link file


        # also make special exp.cmd file for test explosions
        self.logger,info('Making explosion.cmd file.')
        cmdfilepath = os.path.join(expdir, 'explosion.cmd')
        with open(cmdfilepath) as cmdfile:
            cmdfile.write("""
p 16 100000000
p 18 1000
p 156 10
p 390 0
p 437 0
p 536 0
""")
        self.close_logger()



# ; envel dump needs to be earlier for Z=0 stars
# IF composition EQ 'zero' THEN BEGIN
#     genline,'c'
#     genline,'c special Z=0 early execution of the #envel command'
#     if mass LT 15 THEN BEGIN
#         genline,'p 345 500.'
#     ENDIF ELSE if mass LT 20 THEN BEGIN
#         genline,'p 345 1000.'
#     ENDIF ELSE BEGIN
#         genline,'p 345 2000.'
#     ENDELSE
# ENDIF



if __name__ == "__main__":
# add parameters

     import sys
     parm = sys.argv


     MakeLink()
     logging.shutdown()
