#! /usr/bin/python
"""
Python replacement for exp.cpp explosion finder.
"""





import sys
import os
import math
import glob
import numpy as np
import configparser
import io
import logging
import utils
import datetime

import physconst

from kepdump import loaddump
from human import time2human


class explosion(object):
    pattern = "xxx"

    default_config = """
[DEFAULT]
program = ../k
ext = #presn
dump = ../%(base)s%(ext)s
template = explosion.link
logfile = explosion.log
cmdfile = explosion.cmd
logall = True
verbose = True
alpha = 1.0
precision = 2
force = False
run = True
"""

    class State(object):
        def __init__(self, alpha):
            self.val_low  = alpha
            self.val_hi   = alpha
            self.xval_low = -1.e99
            self.xval_hi  = -1.e99

        def initialize(self, xval):
            if self.xval_low < 0:
                self.xval_low = xval
            if self.xval_hi < 0:
                self.xval_hi = xval

    class Result(object):
        def __init__(self,
                     mni = None,
                     ekin = None,
                     mass = None,
                     zone = None):
            self.mni = mni
            self.ekin = ekin
            self.mass = mass
            self.zone = zone

    def alpha2string(self, val):
        return ("{:0."+str(self.alpha_accuracy)+"f}")\
               .format(val / 10**self.alpha_accuracy)

    def generator(self, alpha, link):
        alpha_string = self.alpha2string(alpha)
        link_file_name = link + ".link"
        with open(self.generator_tempelete,"r") as fin,\
                 open(link_file_name,"w") as fout:
            for line in fin:
                fout.write(line.replace(self.pattern,alpha_string))

    def make_cmd(self, link):
        # also copy command file at the end of cmd file if it exists
        cmd_file = self.base_name + link + ".cmd"
        with open(cmd_file,"w") as fcmd:
            fcmd.write("link {}.link\n".format(link))
            if os.path.exists(self.exp_cmd_file):
                with open(self.exp_cmd_file,"r") as fexpcmd:
                    fcmd.write(\
"""
c --- commands from exp.link
""")
                    for line in fexpcmd:
                        fcmd.write(line)

    def get_result(self, link, dump):
        dump_file = self.base_name + link
        if (dump == 0):
            dump_file += "#envel"
        else:
            dump_file += "#final"
        if not os.path.exists(dump_file):
            return None
        dump = loaddump(dump_file)
        xbind = dump.xbind
        rn = dump.rn
        un = dump.un
        uesc = dump.uesc
        # zones = np.argwhere((xbind > 0.)
        #                     * (rn > 1.e10)
        #                     * (un > 0.01 * uesc))
        # if len(zones) > 0:
        #     zone = zones[0][0]
        zones, = np.where((xbind > 0.)
                          * (rn > 1.e10)
                          * (un > 0.01 * uesc))
        if len(zones) > 0:
            zone = zones[0]
        else:
            zone = 1
        mass = dump.zm[zone-1] * physconst.Kepler.solmassi
        ekin = dump.qparm.enk
        mni = (np.sum(dump.xm[zone:-1] * dump.ni56[zone:-1])
               * physconst.Kepler.solmassi)
        return self.Result(mni = mni,
                           ekin = ekin,
                           zone = zone,
                           mass = mass)

    def linkplus(self, link):
        val = 0
        for b in bytearray(link):
            val *= 51
            # g is not allowd
            # z is not allowed
            # a should be 1 not 0 other wise a = aa, ab = b, ...
            if 96 < b < 103:
                val += b - 96
            elif 103 < b < 122:
                val += b - 97
            elif 64 < b < 91:
                val += b - 40
            else:
                self.logger.error("linkplus error with '{}'".format(link))
                raise KeyError(" [LINKPLUS] error: " + link)
        self.logger.debug("val = {}".format(val))
        val += 1
        i = 1
        while val >= 51**i:
            if divmod(val,51**i)[1] < 51**(i-1):
                val += 51**(i-1)
            i += 1
        link = ''
        while val > 0:
            val, c = divmod(val, 51)
            if c > 24:
                link = chr(c + 40) + link
            elif c < 7:
                link = chr(c + 96) + link
            else:
                link = chr(c + 97) + link
        return link


    def nextalpha(self, state, val, xval, xval_goal, xval_use, direction):
        fac = 1.5
        fac_max = 10./3.;
        fac_ext = 4./3.;
        val_good = 0.75;

        logger = logging.getLogger(self.logger.name + '.' + 'nextalpha')

        if (direction == 0):
            logger.info("done: direction = {}".format(direction))
            return True, Ellipsis

        # initialize
        state.initialize(xval)

        # cases...
        if direction < 0:
            if state.val_low == val:
                valx = val / fac
                state.xval_low = xval
                if state.val_low != state.val_hi:
                    dx = state.xval_hi - state.xval_low
                    if dx > 0:
                        logger.info("using extrapolation")
                        valx = (xval_goal - state.xval_low) * (state.val_hi - state.val_low) / dx + state.val_low
                        if valx > val / fac:
                            valx = val / fac
                            logger.info("limiting extrapolation")
                        if valx < val / fac_max:
                            valx = val / fac_max
                            logger.info("limiting extrapolation")
                state.val_hi = val
                state.xval_hi = xval
                val = int(math.ceil(valx))
                if val >= state.val_low:
                    val -= 1
                if val <= 0:
                    logger.info("done: state.val_low = {}".format(state.val_low))
                    return True, val
                state.val_low = val
                state.xval_low = xval
                return False, val
            else:
                state.val_hi = val
                state.xval_hi = xval
                if state.val_hi == state.val_low + 1:
                    logger.info("done: state.val_low + 1 = state.val_hi = {}".format(state.val_hi))
                    return True, val
                dx = state.xval_hi - state.xval_low
                if (dx/state.xval_hi < xval_use) and (dx > 0.):
                    logger.info("using interpolation")
                    valx = (xval_goal - state.xval_low) * (state.val_hi - state.val_low) / dx + state.val_low
                    # if valx < 0.5 * (state.val_low + state.val_hi):
                    if valx - math.floor(valx) > 0.5:
                        val = int(math.ceil(valx))
                    else:
                        val = int(math.floor(valx))
                    if val <= state.val_low:
                        val = state.val_low + 1
                    if (val >= state.val_hi):
                        val = state.val_hi - 1
                    # terminate if last estimate (state.val_hi) was good enough
                    if val == state.val_hi - 1:
                        xval1 = ((val - state.val_low) // (state.val_hi - state.val_low)) * dx + state.xval_low
                        if  (xval - xval_goal  >= 0) and\
                                (xval_goal - xval1 >= 0) and\
                                ((xval - xval_goal) < val_good * (xval_goal - xval1)):
                            val = state.val_hi
                            logger.info("done: good val = {}".format(val))
                            return True, val
                else:
                    val = int((state.val_low + state.val_hi) / 2)
        else:
            if state.val_hi == val:
                valx = val * fac
                state.xval_hi = xval
                if state.val_low != state.val_hi:
                    dx = state.xval_hi - state.xval_low
                    if dx > 0:
                        logger.info("using extrapolation")
                        valx = (xval_goal - state.xval_low) * (state.val_hi - state.val_low) * fac_ext / dx + state.val_low
                        if valx < val * fac:
                            valx = val * fac
                            logger.info("limiting extrapolation")
                        if valx > val * fac_max:
                            valx  =val * fac_max
                            logger.info("limiting extrapolation")
                state.val_low = val
                state.xval_low = xval
                val = int(math.floor(valx))
                if val == state.val_hi:
                    val += 1
                state.val_hi = val
                state.xval_hi = xval
                return False, val
            else:
                state.val_low = val
                state.xval_low = xval
                if state.val_hi == state.val_low + 1:
                    logger.info("done: state.val_low + 1 = state.val_hi = {}".format(state.val_hi))
                    return True, val
                dx = state.xval_hi - state.xval_low;
                if (dx / state.xval_hi < xval_use) and (dx > 0):
                    logger.info("using interpolation")
                    valx = (xval_goal - state.xval_low) * (state.val_hi - state.val_low) / dx + state.val_low
                    # if valx < 0.5 * (state.val_low + state.val_hi)
                    if valx - math.floor(valx) > 0.5:
                        val = int(math.ceil(valx))
                    else:
                        val = int(math.floor(valx))
                    if (val <= state.val_low):
                        val = state.val_low + 1
                    if (val >= state.val_hi):
                        val = state.val_hi - 1
                    # terminate if last estimate (state.val_low) was good enough
                    if val == state.val_low+1:
                        xval1 = ((val - state.val_low) // (state.val_hi - state.val_low)) * dx + state.xval_low
                        if  (xval_goal-xval  >= 0) and\
                                (xval1 - xval_goal >= 0) and\
                                ((xval_goal - xval) < val_good * (xval1 - xval_goal)):
                            val = state.val_low;
                            logger.info("done: good val = {}".format(val))
                            return True, val
                else:
                    val = int((state.val_low + state.val_hi) / 2)
        if val <= state.val_low:
                val = state.val_low + 1;
        if val >= state.val_hi:
                val = state.val_hi - 1;
        return False, val


    def result(self, alpha, link, result):
        temp_link_name = "xxx" + link + ".link"
        link_file_name = link + ".link"
        alpha_string = self.alpha2string(alpha)
        with open(link_file_name,'r') as fin,\
                open(temp_link_name,'w') as fout:
            fout.write("""c # version 1.00.00
c # automatically generated comment
c # ---------------------------------
c #  RESULT:
c # ---------------------------------
c #  alpha:    {:s}
c #  mni:      {:12.5E}
c #  ekin:     {:12.5E}
c #  mass cut: {:8.6F}
c #  zone cut: {:d}
c # ---------------------------------
c # original file follows below this line
""".format(alpha_string,
               result.mni,
               result.ekin,
               result.mass,
               result.zone))
            for line in fin:
                fout.write(line)
        os.remove(link_file_name)
        os.rename(temp_link_name,link_file_name)

    def __init__(self, config_file = 'explosion.cfg'):
        """
        Initialize explosion from config file explosion.cfg.

        Actual values need to be in the [explosion] section, the
        [DEFAULT] section is a backup.  It is also hard-coded, so you
        don't need to overwrite it unless it is being chnaged.  The
        current defaults are in this class in attribute
        'default_config'.

        [DEFAULT]
        program = ../k
        ext = #presn
        dump = ../%(base)s%(ext)s
        template = explosion.link
        logfile = explosion.log
        cmdfile = explosion.cmd
        logall = True
        verbose = True
        alpha = 1.0
        precision = 2
        force = False
        run = True

        [Here is what you *have* to add:]
        link = <link sentinal like "Da">
        ekin_or_mni = <value>
        base = <run name>

        [example]
        link = Da
        alpha = 1.0
        ekin = 1.2e+51
        base = s25

        [explosion]
        link = Pa
        alpha = 1.0
        ekin = 1.2e+51
        base = u25

        ======

        So, with the defaults, the files that you need in the
        explosion directory are just

        explosion.link
        explosion.cmd
        explosion.cfg

        """

        start_time = datetime.datetime.now()

        config = configparser.SafeConfigParser()
        config.readfp(io.BytesIO(self.default_config))
        config.read(config_file)

        section = 'explosion'
        run = config.getboolean(section,'run')
        force = config.getboolean(section,'force')
        kepler = config.get(section,'program')
        presn_dump_name = config.get(section,'dump')
        self.base_name = config.get(section,'base')
        self.generator_tempelete = config.get(section,'template')
        generator_start = config.get(section,'link')
        alpha_start = config.getfloat(section,'alpha')
        self.alpha_accuracy = config.getint(section,'precision')
        if config.has_option(section,'mni'):
            flag = 'mni'
        else:
            flag = 'ekin'
        val = config.getfloat(section,flag)
        logfile = config.get(section,'logfile')
        self.exp_cmd_file = config.get(section,'cmdfile')
        verbose = config.getboolean(section,'verbose')
        logall = config.getboolean(section,'logall')

        # set up log output
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, 'w')
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.WARNING
        fh.setLevel(level)
        formatter = utils.UTCFormatter('%(asctime)s%(msecs)d [%(name)s] %(levelname)s: %(message)s',
                                      datefmt = '%Y%m%d%H%M%S')
        fh.setFormatter(formatter)
        root_logger = logging.getLogger('')
        if logall and len(root_logger.handlers) == 0:
            root_logger.addHandler(fh)
            root_logger.setLevel(level)
        else:
            self.logger.addHandler(fh)

        # init alpha iterations
        alpha = int(alpha_start * 10**self.alpha_accuracy)
        state = self.State(alpha)

        link = generator_start
        alpha_best = -1
        link_best = "#"
        best = 1.e99;
        zone_best = 0
        mass_best = 0

        self.logger.info("{}: first alpha = {}".format(link, alpha))

        # set up limits for mod. reg. falsi algorithm for solution finder
        if flag == "mni":
            val_use = 0.2
        else:
            val_use = 10.

        # start of main loop
        done = False
        while not done:
            if run:
                finished = True
                dump_file_name = self.base_name + link +"#final"
                if not os.path.exists(dump_file_name):
                    finished = False
                dump_file_name = self.base_name + link +"#envel"
                if not os.path.exists(dump_file_name):
                    finished = False
                if (finished == False) or (force == True):
                    call_string = "{} {}{} ".format(kepler, self.base_name, link)

                    # look if a restart (z) dump is present
                    dump_file_name = self.base_name + link + "z"
                    if (not os.path.exists(dump_file_name)) or (force == True):
                        self.logger.info("RUN")
                        self.generator(alpha, link)
                        self.make_cmd(link);
                        call_string += presn_dump_name
                    else:
                        self.logger.info("CONTINUE")
                        call_string += "z"
                    call_string += " k </dev/null >/dev/null"
                    self.logger.info("CALL: {}".format(call_string))
                    os.system(call_string)
                else:
                    self.logger.info("FINISHED")

            # analyze the result
            envel = self.get_result(link, 0)
            if envel == None:
                self.logger.error("envel fallback determination")
                return None
            self.logger.info("Fallback up to zone = {} with mass = {}".format(
                envel.zone,
                envel.mass))
            final = self.get_result(link, 1);
            if final == None:
                self.logger.error("final fallback determination")
                return None
            if (envel.zone < final.zone):
                self.logger.warning("LATE FALLBACK")
                self.logger.warning("Fallback up to zone = {} with mass = {}".format(
                    final.zone,
                    final.mass))
                result = final
            else:
                result = envel
                result.ekin = final.ekin
            self.logger.info("mni = {}, ekin = {}".format(result.mni, result.ekin))
            self.result(alpha, link, result)

            # determine next piston value
            if flag == "mni":
                val_current = result.mni
            else:
                val_current = result.ekin
            direction = cmp(val, val_current)
            dist = direction * (val - val_current)
            if (dist < best):
                best = dist
                val_best = val_current
                alpha_best = alpha
                link_best = link
                zone_best = result.zone
                mass_best = result.mass
            self.logger.info("next step: {},{},{}".format(val,
                val_current,
                direction))
            done, alpha = self.nextalpha(
                state,
                alpha,
                val_current,
                val,
                val_use,
                direction)
            if not done:
                link = self.linkplus(link);
                self.logger.info("{}: new alpha = {} ({},{})".format(
                    link,
                    alpha,
                    state.val_low,
                    state.val_hi))

        end_time = datetime.datetime.now()
        load_time = end_time - start_time
        self.logger.info('finished in ' + time2human(load_time.total_seconds()))

        call_string = self.alpha2string(alpha_best)
        self.logger.critical("""

---------------------------------
 RESULT:
---------------------------------
 goal:       {0:4s} = {1:12.5E}
 best value: {0:4s} = {2:12.5E}
 deviation:  {0:4s} = {3:12.5E}
 best alpha: {4:s}
 in link:    {5:s}.link
 mass cut:   {6:8.6F}
 zone cut:   {7:d}
---------------------------------


done.""".format(
            flag,
            val,
            val_best,
            best,
            call_string,
            link_best,
            mass_best,
            zone_best))

        # clean up
        for filename in glob.iglob("xxx*"):
            os.remove(filename)

if __name__ == "__main__":
    explosion()
    logging.shutdown()
