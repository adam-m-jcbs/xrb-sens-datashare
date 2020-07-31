#! /bin/env python3
"""
Python replacement for exp.cpp explosion finder.
"""

import sys
import os
import math
import glob
import numpy as np
from configparser import SafeConfigParser as Config
import io
import logging
import utils
import datetime
import subprocess

import physconst
from utils import float2str
from logged import Logged
from kepdump import _load as loaddump
from human import time2human
from human import version2human

def link2val(link):
    """
    Convert link letter sequence into numerical value.
    """
    val = 0
    for b in bytearray(link, encoding='ASCII'):
        val = val * 50
        # g is not allowed
        # z is not allowed
        # a should be 1 not 0 other wise a = aa, ab = b, ...
        if 96 < b < 103:
            val += b - 96
        elif 103 < b < 122:
            val += b - 97
        elif 64 < b < 91:
            val += b - 40
        else:
            raise KeyError(" [LINK2VAL] error: " + link)
    return val

def num2link(val):
    """
    Convert numerical sequence index into link letter sequence.
    """
    link = ''
    while val > 0:
        val, c = divmod(val-1, 50)
        if c > 23:
            link = chr(c + 41) + link
        elif c < 6:
            link = chr(c + 97) + link
        else:
            link = chr(c + 98) + link
    return link

def linkplus(link):
    """
    Return next letter sequence.
    """
    return num2link(link2val(link) + 1)

class Link(object):
    def __init__(self, val):
        if isinstance(val, str):
            self.val = link2val(val)
        else:
            self.val = val
    def name(self):
        return num2link(self.val)
    def __str__(self):
        return self.name()
    __repr__ = __str__
    def __format__(self, fmt):
        return format(str(self), fmt)
    def __add__(self, other):
        if isinstance(other, str):
            return self.name() + other
        return self.__class__(self.val + other)
    def __radd__(self, other):
        if isinstance(other, str):
            return other + self.name()
        return self.__class__(self.val + other)

    def __sub__(self, other):
        return self.__class__(self.val - other)
    __rsub__ = __sub__

class State(Logged):
    """
    Current state of explosion.
    Usually should have 2 Results.

    Handle alpha as in with #precision digits after .
    """
    def __init__(self,
                 config = None,
                 # all for _from_results
                 results = None,
                 flag = None,
                 goal = None,
                 precision = 2):

        self.best = None

        if config is not None:
            self._from_config(config)
        else:
            self._from_results(
                results = results,
                precision = precision,
                goal = goal,
                flag = flag)
        # set up limits for Modified Regular Falsi Algorithm
        # for solution finder
        if self.flag == 'mni':
            self.limit = 0.2
        else:
            self.limit = 10.

        if self.hi.zone == -1 and self.lo.zone != -1:
            self.hi = self.lo
        if self.lo.zone == -1 and self.hi.zone != -1:
            self.lo = self.hi


    def _from_results(self,
                      results = None,
                      precision = 2,
                      flag = None,
                      goal = None):
        self.precision = precision
        self.flag = flag
        self.goal = goal
        if len(results) == 1:
            self.lo = self.hi = results[0]
            return

        assert len(results) == 2
        if results[0].alpha < results[1].alpha:
            (self.lo, self.hi) = results
        else:
            (self.hi, self.lo) = results


    def _from_config(self, config):
        section = 'explosion'
        self.alpha = config.getfloat(section, 'alpha')
        self.precision = config.getfloat(section, 'precision')
        self.basename = config.get(section,'base')
        # self.alpha = int(round(self.alpha * 10**self.precision))
        if config.has_option(section, 'mni'):
            self.flag = 'mni'
        else:
            self.flag = 'ekin'
        self.goal = config.getfloat(section, self.flag)

        section = 'min'
        if config.has_section(section):
            self.lo = Result(
                config = config,
                section = section)
        else:
            self.lo = Result(
                alpha = self.alpha)

        section = 'max'
        if config.has_section(section):
            self.hi = Result(
                config = config,
                section = section)
        else:
            self.hi = Result(
                alpha = self.alpha)


    def update(self,
               link = None,
               # optional
               basename = None,
               alpha = None,
               precision = None,
               silent = False):
        """
        Compute new result from link and return next alpha.
        """
        self.setup_logger(silent)
        precision_save = self.precision
        if precision is None:
            precision = self.precision
        self.precision = precision

        if alpha is None:
            alpha = self.alpha
        else:
            alpha = int(alpha * 10**self.precision)
        if basename is None:
            basename = self.basename
        new = Result(
            basename = basename,
            link = link,
            alpha = alpha,
            precision = precision,
            update_link = True,
            silent = silent)
        assert new is not None, 'obtaining new result failed'

        if self.hi.zone is None:
            self.hi = new
        if self.lo.zone is None:
            self.lo = new

        self._update_interval(new)
        self.alpha = self.alpha_forecast()

        self.precsion = precision_save
        self.close_logger()

    def alpha_forecast(self,
                       silent=False):
        """
        Predict and return next alpha based on hi/lo values.
        """
        self.setup_logger(silent)
        alpha = self._next_alpha()
        if alpha == 0:
            if (abs(self.lo_val() - self.goal) <
                abs(self.hi_val() - self.goal)):
                self.best = self.lo
            else:
                self.best = self.hi
            alpha = self.best.alpha
        self.close_logger()
        return alpha

    def val(self, result):
        """
        Return relevant value from result object.
        """
        return result.__dict__[self.flag]

    def eq_alpha(self, alpha):
        """
        Return whether alpha is equal to stored value within precision
        """
        return abs(self.alpha - alpha)*10**self.precision < 0.5

    def lo_val(self):
        """
        Return lo value
        """
        return self.val(self.lo)

    def hi_val(self):
        """
        Return hi value
        """
        return self.val(self.hi)

    def lo_Alpha(self):
        """
        Return lo alpha
        """
        return int(round(self.lo.alpha * 10**self.precision))

    def hi_Alpha(self):
        """
        Return hi alpha
        """
        return int(round(self.hi.alpha * 10**self.precision))

    def dval(self):
        """
        Return hi value
        """
        return self.hi_val() - self.lo_val()

    def dAlpha(self):
        """
        Return hi value
        """
        return self.hi_Alpha() - self.lo_Alpha()

    def _update_interval(self, new):

        if self.val(new) < self.goal:
            if self.hi_val() < self.goal:
                self.lo = self.hi
                self.hi = new
            elif self.lo_val() > self.goal:
                self.hi = self.lo
                self.lo = new
            else:
                self.lo = new
        else:
            if self.lo_val() > self.goal:
                self.hi = self.lo
                self.lo = new
            elif self.hi_val() < self.goal:
                self.lo = self.hi
                self.hi = new
            else:
                self.hi = new

    def _Alpha2alpha(self, Alpha):
        return Alpha * 0.1**self.precision

    def _next_alpha(self):
        """
        Compute and return next value of alpha or 0 if finished.
        """

        # Alpha is the quantized integer in contraxt to the float alpha

        fac = 1.5
        fac_max = 10./3.
        fac_ext = 4./3.
        val_good = 0.75

        if ((self.goal == self.lo_val()) or
            (self.goal == self.hi_val())):
            self.logger.info('Done.')
            return 0

        dAlpha = self.dAlpha()
        dval = self.dval()

        # lo bound
        if self.goal < self.lo_val():
            Alpha = self.lo_Alpha() / fac
            if dAlpha != 0:
                if dval > 0:
                    self.logger.info("using extrapolation")
                    Alpha = ((self.goal - self.lo_val()) *
                             dAlpha * fac_ext / dval + self.lo_Alpha())

                    if Alpha > self.lo_Alpha() / fac:
                        Alpha = self.lo_Alpha() / fac
                        self.logger.info("extending extrapolation")

                    if Alpha < self.lo_Alpha() / fac_max:
                        Alpha = self.lo_Alpha() / fac_max
                        self.logger.info("limiting extrapolation")

            Alpha = int(math.ceil(Alpha))
            if Alpha >= self.lo_Alpha():
                Alpha = self.lo_Alpha() - 1
            if Alpha <= 0:
                Alpha = 1

            if Alpha == self.lo_Alpha():
                # implies Alpha = 1
                self.logger.info('done: state.val_low = {}'.format(
                    self.lo_Alpha()))
                return 0

        # hi bound
        elif self.goal > self.hi_val():
            Alpha = self.hi_Alpha() * fac
            if dAlpha != 0:
                if dval > 0:
                    self.logger.info("using extrapolation")
                    Alpha = ((self.goal - self.hi_val()) *
                             dAlpha / dval + self.hi_Alpha())

                    if Alpha < self.hi_Alpha() * fac:
                        Alpha = self.hi_Alpha() * fac
                        self.logger.info("extending extrapolation")

                    if Alpha > self.hi_Alpha() * fac_max:
                        Alpha = self.hi_Alpha() * fac_max
                        self.logger.info("limiting extrapolation")

            Alpha = int(math.floor(Alpha))
            if Alpha <= self.hi_Alpha():
                Alpha = self.hi_Alpha() + 1

        # interpolation
        else:
            if self.hi_Alpha() - self.lo_Alpha() <= 1:
                self.logger.info(
                        'done: low + 1 = hi = {}'.format(self.hi_Alpha()))
                return 0

            # we stick with linear interpolation for now
            if (dval / self.hi_val() < self.limit) and (dval > 0.):
                self.logger.info("using interpolation")
                Alpha = ((self.goal - self.lo_val()) *
                         dAlpha / dval + self.lo_Alpha())
                # if Alpha < 0.5 * dAlpha:
                if Alpha - math.floor(Alpha) > 0.5:
                    Alpha = int(math.ceil(Alpha))
                else:
                    Alpha = int(math.floor(Alpha))
            else:
                Alpha = int((self.lo_Alpha() + self.hi_Alpha()) / 2)

            if Alpha <= self.lo_Alpha():
                Alpha = self.lo_Alpha() + 1
            if Alpha >= self.hi_Alpha():
                Alpha = self.hi_Alpha() - 1

            # terminate if last estimate (new.val) was good enough
            val1 = ((Alpha - self.lo_Alpha()) / dAlpha) * dval + self.lo_val()
            if ((Alpha == self.hi_Alpha() - 1) or
                (Alpha == self.lo_Alpha() + 1)):
                if (((val1 < self.goal < self.hi_val()) and
                     ((self.hi_val() - self.goal) <
                       val_good * (self.goal - val1))) or
                    ((val1 > self.goal > self.lo_val()) and
                     ((self.goal - self.lo_val()) <
                      val_good * (val1 - self.goal)))):
                    self.logger.info('done: good value.')
                    return 0

        return self._Alpha2alpha(Alpha)

    def save(self,
             configfile = None):
        """
        write state to cfg file
        """
        config = Config()

        section = 'explosion'
        config.add_section(section)
        config.set(section,'alpha'    ,'{:g}'.format(self.alpha))
        config.set(section,'precision','{:g}'.format(self.precision))
        config.set(section,'limit'    ,'{:g}'.format(self.limit))
        config.set(section,'goal'     ,'{:g}'.format(self.goal))
        config.set(section,'flag'     ,'{:s}'.format(self.flag))
        config.set(section,'base'     ,'{:s}'.format(self.basename))

        if self.best is not None:
            self.best.add_to_config(
                config = config,
                section = 'best')
        if self.hi is not None:
            self.hi.add_to_config(
                config = config,
                section = 'hi')
        if self.lo is not None:
            self.lo.add_to_config(
                config = config,
                section = 'lo')

        with open(configfile,'w') as f:
            config.write(f)


class Result(Logged):
    """
    Hold result of a test explosion.
    """
    def __init__(self,
                 basename = None,
                 link = None,
                 linkfile = None,
                 alpha = None,
                 precision = 2,
                 config = None,
                 section = None,
                 update_link = False,
                 from_link = False,
                 silent = False):

        self.setup_logger(silent)
        self.alpha = alpha
        if from_link:
            self._from_link(
                linkfile = linkfile,
                link = link)
        elif config is not None:
            self._from_config(
                config = config,
                section = section)
        elif link is not None:
            self._from_dump(
                basename = basename,
                link = link)
            if update_link:
                self._to_link()
        else:
            self._from_void()

        self.close_logger()

    def val(self, flag):
        """
        Return selected result value.
        """
        if flag == 'mni':
            return self.mni
        return self.ekin


    def _from_config(self,
                config = None,
                section = None):
        self.mni   = config.getfloat(section, 'mni'  )
        self.ekin  = config.getfloat(section, 'ekin' )
        self.mass  = config.getfloat(section, 'mass' )
        self.zone  = config.getint  (section, 'zone' )
        self.mpist = config.getfloat(section, 'mpist')
        self.link  = config.get     (section, 'link' )
        self.alpha = config.getfloat(section, 'alpha')

    def add_to_config(self,
                config = None,
                section = None):
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, 'mni'  , '{:g}'.format(self.mni  ))
        config.set(section, 'ekin' , '{:g}'.format(self.ekin ))
        config.set(section, 'mass' , '{:g}'.format(self.mass ))
        config.set(section, 'zone' , '{:d}'.format(self.zone ))
        config.set(section, 'mpist', '{:g}'.format(self.mpist))
        config.set(section, 'link' , '{!s}'.format(self.link ))
        config.set(section, 'alpha', '{:g}'.format(self.alpha))

    def _from_void(self):
        self.mni   = None
        self.ekin  = None
        self.mass  = None
        self.zone  = None
        self.mpist = None
        self.link  = None

    version = 20000
    version_string = version2human(version)
    version_line = 'c # Version ' + version_string

    def _from_link(self,
                   linkfile = None,
                   link = None):
        if linkfile is None:
            self.link = link
            linkfile = self.link + '.link'
        else:
            if link is None:
                self.link = Link((os.path.split(linkfile))[1].rsplit('.',1)[0])
        with open(linkfile,'r') as f:
            content = f.readlines()
            if content[0].startswith(self.version_line):
                self.alpha = float(content[ 5][14:])
                self.mni   = float(content[ 6][14:])
                self.ekin  = float(content[ 7][14:])
                self.mass  = float(content[ 8][14:])
                self.zone  =   int(content[ 9][14:])
                self.mpist = float(content[10][14:])
            elif content[0].startswith('c # version 1.00.00'):
                self.alpha = float(content[ 5][14:])
                self.mni   = float(content[ 6][14:])
                self.ekin  = float(content[ 7][14:])
                self.mass  = float(content[ 8][14:])
                self.zone  =   int(content[ 9][14:])
                if self.zone == 1:
                    self.mpist = self.mass
                else:
                    self.mpist = None
                    self.ipist = None
                    for line in content[10:]:
                        if line.startswith('bounce '):
                            self.ipist = int(line[7:10])
                            break
            else:
                self.alpha = None
                self.mni   = None
                self.ekin  = None
                self.mass  = None
                self.zone  = None
                self.mpist = None
                self.logger.debug('could not load {:s}'.format(linkfile))

    def _from_dump(self,
                   basename = None,
                   link = None):

        run_name = basename + link

        dump_names = ('nucleo', 'envel', 'final')
        results = {}
        for d in dump_names:
            dump_name = run_name + '#' + d
            dump = loaddump(dump_name)
            if dump is None:
                self.logger.error('{} fallback determination'.format(d))
                raise Exception('Dump not found: {}'.format(dump_name))
            results[d] = dump.sn_result()

        self.logger.info("Fallback up to zone = {} with mass = {}".format(
            results['envel']['zone'],
            results['envel']['mass']))
        if (results['envel']['zone'] < results['final']['zone']):
            self.late = True
            self.logger.warning("LATE FALLBACK")
            self.logger.warning("Fallback up to zone = {} with mass = {}".format(
                results['final']['zone'],
                results['final']['mass']))
            result = results['final']
        else:
            result = results['envel']
            result['ekin'] = results['final']['ekin']
            self.late = False
        # TODO - check for fallback in #nucleo

        self.mni   = result['mni']
        self.ekin  = result['ekin']
        self.mass  = result['mass']
        self.zone  = result['zone']
        self.mpist = result['mpist']
        self.link  = link

        self.logger.info('mni = {}, ekin = {}'.format(
                self.mni, self.ekin))

    header_end = 'c # original file follows below this line'

    link_header_format = version_line + """
c # automatically generated comment
c # ---------------------------------
c #  RESULT:
c # ---------------------------------
c #  alpha:    {:s}
c #  mni:      {:12.6e}
c #  ekin:     {:12.6e}
c #  mass cut: {:12.6e}
c #  zone cut: {:d}
c #  piston:   {:12.6e}
c # ---------------------------------
""" + header_end + '\n'

    def _to_link(self):
        linkfile = self.link + '.link'
        with open(linkfile,'r') as f:
            content = f.readlines()
        with open(linkfile,'w') as f:
            f.write(self.link_header_format.format(
                    float2str(self.alpha),
                    self.mni,
                    self.ekin,
                    self.mass,
                    self.zone,
                    self.mpist))
            for i,line in enumerate(content):
                if not line.startswith('c #'):
                    break
                if line.startswith(self.header_end):
                    del content[0:i+1]
                    break
            for line in content:
                f.write(line)


class Explosion(object):
    """
    Explosion main program "class".
    """
    sentinel = "xxx"

#     default_config = """
# [DEFAULT]
# program = ../k
# ext = #presn
# dump = ../%(base)s%(ext)s
# template = explosion.link
# logfile = explosion.log
# cmdfile = explosion.cmd
# logall = True
# verbose = True
# alpha = 1.0
# precision = 2
# force = False
# run = True
# """

    default_config = dict(
        program = '../k',
        ext = '#presn',
        dump = '../%(base)s%(ext)s',
        template = 'explosion.link',
        logfile = 'explosion.log',
        cmdfile = 'explosion.cmd',
        logall = 'True',
        verbose = 'True',
        alpha = '1.0',
        precision = '2',
        force = 'False',
        run = 'True')

    out_format = """

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


done."""

    def generator(self, alpha, link):
        alpha_string = float2str(alpha)
        link_file_name = link + ".link"
        with open(self.generator_tempelete,"r") as fin,\
            open(link_file_name,"w") as fout:
            for line in fin:
                fout.write(line.replace(self.sentinel,alpha_string))

    def make_cmd(self, link):
        # also copy command file at the end of cmd file if it exists
        cmd_file = self.base_name + link + ".cmd"
        with open(cmd_file,"w") as fcmd:
            fcmd.write("link {}.link\n".format(link))
            if os.path.exists(self.exp_cmd_file):
                with open(self.exp_cmd_file,"r") as fexpcmd:
                    fcmd.write('\nc --- commands from exp.link\n')
                    for line in fexpcmd:
                        fcmd.write(line)

    def existing_alpha(self, link):
        with open(link + '.link',"r") as f:
            for line in f:
                if line.startswith('bounce '):
                    return float(line.split()[5])

    def __init__(self, config_file = 'explosion.cfg'):
        """
        Initialize explosion from config file explosion.cfg.

        Actual values need to be in the [explosion] section, the
        [DEFAULT] section is a backup.  It is also hard-coded, so you
        don't need to overwrite it unless it is being changed.  The
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

        config = Config(self.default_config)
#        config.readfp(io.BytesIO(self.default_config))
        config.read(config_file)

        section = 'explosion'
        run = config.getboolean(section,'run')
        force = config.getboolean(section,'force')
        kepler = config.get(section,'program')
        presn_dump_name = config.get(section,'dump')
        self.base_name = config.get(section,'base')
        self.generator_tempelete = config.get(section,'template')
        generator_start = config.get(section,'link')

        logfile = config.get(section,'logfile')
        self.exp_cmd_file = config.get(section,'cmdfile')
        verbose = config.getboolean(section,'verbose')
        logall = config.getboolean(section,'logall')

        # set up log output
        # maybe this this be replaced by
        # deriving class from Logged and
        # self.setup_logger(silent = not verbose, logfile=logfile, format='UTC')
        # and at the end: self.close_logger()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, 'w')
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.WARNING
        fh.setLevel(level)
        formatter = utils.UTCFormatter('%(asctime)s%(msecs)03d %(nameb)-12s %(levelname)s: %(message)s',
                                      datefmt = '%Y%m%d%H%M%S')
        fh.setFormatter(formatter)
        root_logger = logging.getLogger('')
        if logall and len(root_logger.handlers) == 0:
            root_logger.addHandler(fh)
            root_logger.setLevel(level)
        else:
            self.logger.addHandler(fh)

        # set up state for explosion
        state = State(
            config = config)

        link = Link(generator_start)

        self.logger.info("{:s}: first alpha = {:g}".format(link, state.alpha))

        while state.best is None:
            if run:
                finished = True
                dump_file_name = self.base_name + link + "#final"
                if not os.path.exists(dump_file_name):
                    finished = False
                dump_file_name = self.base_name + link + "#envel"
                if not os.path.exists(dump_file_name):
                    finished = False
                # check if run parameters are identical
                if finished and not force:
                    if not state.eq_alpha(self.existing_alpha(link)):
                        self.logger.warning('Previous run '+link+' had different alpha.  RERUNNING.')
                        finished = False
                        force = True
                        os.remove(self.base_name + link + "#final")
                        os.remove(self.base_name + link + "#envel")
                        os.remove(self.base_name + link + "z")
                        os.remove(link + ".link")
                if (not finished) or force:
                    # call_string = "{} {}{} ".format(kepler, self.base_name, link)

                    # # look if a restart (z) dump is present
                    # dump_file_name = self.base_name + link + "z"
                    # if (not os.path.exists(dump_file_name)) or force:
                    #     self.logger.info("RUN")
                    #     self.generator(state.alpha, link)
                    #     self.make_cmd(link);
                    #     call_string += presn_dump_name
                    # else:
                    #     self.logger.info("CONTINUE")
                    #     call_string += "z"
                    # call_string += " k </dev/null >/dev/null"
                    # self.logger.info("CALL: {}".format(call_string))
                    # os.system(call_string)

                    args = [kepler, self.base_name + link]

                    # look if a restart (z) dump is present
                    dump_file_name = self.base_name + link + "z"
                    if (not os.path.exists(dump_file_name)) or force:
                        self.logger.info("RUN")
                        self.generator(state.alpha, link)
                        self.make_cmd(link);
                        args += [presn_dump_name]
                    else:
                        self.logger.info("CONTINUE")
                        args += ['z']
                    args += ['k']
                    self.logger.info("CALL: {}".format(' '.join(args)))
                    with open(os.devnull,'r') as null_in:
                        with open(os.devnull,'w') as null_out:
                            subprocess.call(args,
                                            shell  = False,
                                            stdin  = null_in,
                                            stdout = null_out,
                                            stderr = subprocess.STDOUT)
                else:
                    self.logger.info("FINISHED")

            state.update(link)

            if state.best is None:
                link += 1
                self.logger.info("{}: new alpha = {} ({},{})".format(
                    link,
                    state.alpha,
                    state.lo_val(),
                    state.hi_val()))

        end_time = datetime.datetime.now()
        load_time = end_time - start_time
        self.logger.info('finished in ' + time2human(load_time.total_seconds()))

        self.logger.critical(self.out_format.format(
            state.flag,
            state.goal,
            state.val(state.best),
            abs(state.goal - state.val(state.best)),
            float2str(state.best.alpha),
            state.best.link,
            state.best.mass,
            state.best.zone))

        state.save('explosion.res')

        # clean up
        for filename in glob.iglob("xxx*"):
            os.remove(filename)

if __name__ == "__main__":
    Explosion()
    logging.shutdown()
