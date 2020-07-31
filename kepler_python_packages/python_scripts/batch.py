#! /bin/env python3
# ls /proc/33613/cwd | cut -d ">" -f 2 | cut -d" " -f2

"""
Module to replace batch system
"""

import os
import os.path
import socket
import datetime
import string
import time
import numpy as np
import sys
import shutil
import glob
import re
import subprocess
import contextlib
from collections import OrderedDict

import psutil, logdata, convdata, winddata

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    print(' [YAML] Install LibYAML!')
    print(' [YAML] requires (yum) install of libyaml-devel before (pip3) install of pyyaml.')

import physconst
import kepdump

from logged import Logged
from kepgen import KepEnv
from human import time2human
from human import byte2human
from utils import touch, iterable
from fortranfile import FortranReader
from fortranfile.errors import RecLenError

default_path = '/c/alex/kepler/prog3d'

def Load(f):
    return load(f, Loader=Loader)

class Config(Logged):
    def __init__(self,
                 configfile = KepEnv(silent=True).batchfile):
        with open(configfile) as f:
            self.tasks = Load(f)

    def another(self):
        config = SafeConfigParser(allow_no_value = True)
        with open(configfile,'r') as f:
            config.readfp(f)
        section = 'runs'
        directories = config.options(section)
        section = 'global'
        default = dict(config.items(section))
        tasks = dict()
        for d in directories:
            tasks[d] = dict(default)
            if config.has_section(d):
                items = config.items(d)
                tasks[d].update(dict(items))
                option = 'dir'
                if config.has_option(d, option):
                    dir = config.get(d, option)
                else:
                    dir = d
                tasks[d][option] = dir
        self.tasks = tasks


class Run(Logged):
    def __init__(self):
        pass

# likely there should be a factory function that returns different
# KEPLER Run classes, e.g., for explosions, testexplosions, main runs,
# XRB, ... ?

# maybe should be a base class that derives different run types -
# presn, estexp, explosion, xrb


class KepRun(Logged):
    """
    class for KEPLER process
    """
    def __init__(self,
                 process = None,
                 pid = None,
                 cwd = None,
                 run = None,
                 active = None,
                 ):
        """
        Kepler Run object

        active can be
         * None: check
         * True: must run
         * False: must not run
         * 'ignore': don't check - usful for fast object creation
        """
        if process is None:
            if pid is not None:
                process = psutil.Process(pid)
            elif active != 'ignore':
                p = []
                for r in running():
                    ok = True
                    if cwd is not None:
                        if not os.path.samefile(r.cwd(), cwd):
                            ok = False
                    if ok and run is not None:
                        if r.cmdline()[1] != run:
                            ok = False
                    if ok:
                        p += [r]
                if len(p) > 1:
                    raise Exception('Multiple matches found')
                if active is True:
                    if len(p) == 0:
                        raise Exception('No match found')
                    process = p[0]
                elif active is None:
                    if len(p) != 1:
                        process = None
                    else:
                        process = p[0]
                elif active is False:
                    if len(p) != 0:
                        raise Exception('Process is running')
                    else:
                        process = None
                else:
                    raise Exception('Unknown option for \'active\': "{}"'.format(active))
        if active is True and process is None:
            raise Exception('Run is not active')
        self.process = process
        if process is not None:
            try:
                self.cwd = process.cwd()
            except:
                self.cwd = None
            if cwd is not None:
                assert self.cwd == cwd
            self.run = process.cmdline()[1]
            if run is not None:
                assert self.run == run
            self.start = process.cmdline()[2]
            if self.start == 'g':
                self.start = self.run + self.start
        else:
            assert cwd is not None
            assert os.path.isdir(cwd)
            if run is None:
                run = os.path.split(cwd)[-1]
            self.cwd = cwd
            self.run = run
            self.start = None

    def presn(self):
        f = os.path.join(self.cwd, self.run + '#presn')
        if os.path.isfile(f):
            d = kepdump._load(f)
            return d
        else:
            return None

    def is_restart(self):
        return not self.start.endswith('g')

    def z(self):
        d = kepdump._load(os.path.join(self.cwd, self.run + 'z'))
        return d

    def ifiles(self):
        run = os.path.join(self.cwd, self.run)
        for p in glob.iglob(run + '*'):
            ext = p[len(run):]
            if ext.startswith('#'):
                yield p
            else:
                x = re.findall('^z(?:[1-9][0-9]*)?$', ext)
                if len(x) > 0:
                    yield p

    def lastfile(self):
        dumps = list(self.ifiles())
        t = [os.path.getmtime(d) for d in dumps]
        i = np.argsort(t)
        return dumps[i[-1]]

    def lastdump(self):
        for i in range(10):
            try:
                d = kepdump._load(self.lastfile())
                return d
            except:
                time.sleep(1)

    def cycle(self):
        d = self.lastdump()
        return d.qparm.ncyc

    def is_clean(self):
        return is_clean(self.cwd, self.run)

    def clean(self):
        if self.is_clean():
            return

        run = self.run
        rundir = self.cwd
        self._info('cleaning directory: {}, run: {}'.format(rundir, run))

        del_list = (
            '^xxx.*',
            '^' + run + '#[0-9]+$',
            '^' + run + 'z[0-9]*$',
            '^' + run + '_[0-9]+$',
            '^nofix',
            '^broken',
            '^continue',
            )

        compress_ext = (
            'cnv',
            'wnd',
            'log',
            'lc',
            )

        del_list = [re.compile(i) for i in del_list]
        compress_list = [re.compile('^' + run + r'\.' + i + '$') for i in compress_ext]

        deleted_bytes = 0
        deleted_files = 0
        compressed_bytes = 0
        compressed_files = 0
        kept_bytes = 0
        kept_files = 0
        for fn in glob.glob(os.path.join(rundir, '*')):
            name = os.path.basename(fn)
            stat = os.stat(fn)
            for x in del_list:
                if len(x.findall(name)) > 0:
                    self.logger.info('- removing  {}'.format(fn))
                    os.remove(fn)
                    deleted_bytes += stat.st_size
                    deleted_files += 1
                    break
            else:
                for x in compress_list:
                    if len(x.findall(name)) > 0:
                        self.logger.info('* compressing {}'.format(fn))
                        compress_file(fn)
                        compressed_bytes += stat.st_size
                        compressed_files += 1
                        break
                else:
                    self.logger.info('+ keeping    {}'.format(fn))
                    kept_bytes += stat.st_size
                    kept_files += 1

        self.logger.info("deleted    {:5d} files ({})".format(
            deleted_files,
            byte2human(deleted_bytes),
            ))
        self.logger.info("compressed {:5d} files ({})".format(
            compressed_files,
            byte2human(compressed_bytes),
            ))
        self.logger.info("kept       {:5d} files ({})".format(
            kept_files,
            byte2human(kept_bytes),
            ))

        # sys.exit()

def is_clean(rundir, run):
    base = os.path.join(rundir, run + '.log')
    return ((not os.path.exists(base))
        or os.path.exists(base + '.xz')
        or os.path.exists(base + '.gz')
        or os.path.exists(base + '.bz2')
        )

def compress_file(filename,
                  compressor = 'xz',
                  flags = None,
                  nice = 19,
              ):
    """
    Compress file in background
    """

    _flags = {'xz'   : ('-e'),
              'gzip' : ('-9'),
              'bzip2': ('--best'),
             }

    assert compressor in _flags, 'unknown compressor'
    filename = os.path.expanduser(os.path.expandvars(filename))
    assert os.path.exists(filename), 'file does not exists'
    filepath = os.path.dirname(filename)
    assert os.access(filepath, os.W_OK), 'directory is not writable'
    if flags is None:
        flags = _flags[compressor]

    args = [shutil.which(compressor)]
    args += list(iterable(args))
    args += [filename]

    p = psutil.Popen(args,
                     shell  = False,
                     cwd    = filepath,
                     stdin  = subprocess.DEVNULL,
                     stdout = subprocess.DEVNULL,
                     stderr = subprocess.DEVNULL,
                     start_new_session = True)
    if (nice is not None) and (nice > 0):
        p.nice(nice)

def find_runs(path = default_path):
    runs = []
    for p in glob.iglob(os.path.join(path, '*')):
        if os.path.isdir(p):
            kepler = os.path.exists(os.path.join(p, 'k'))
            run = os.path.split(p)[-1]
            gen = os.path.exists(os.path.join(p, run + 'g'))
            if gen and kepler:
                runs += [run]
    return runs


def running():
    username = os.getlogin()
    jobs = []
    for p in psutil.process_iter():
        if p.username() == 'alex':
            try:
                cwd = p.cwd()
            except:
                cwd = None
            if cwd is not None:
                if cwd.endswith('(deleted)'):
                    print('killing {:s} in {:s}'.format(p.exe(), cwd))
                    p.kill()
                else:
                    if p.name() == 'k' and p.cmdline()[1] != 'xxx':
                        # print("{cwd:s}: {run:s}".format(
                        #     cwd = cwd,
                        #     run = p.cmdline()[1],
                        #     ))
                        jobs += [p]
                        # print("{cwd:s}: {run:s}@{cycle:d}".format(
                        #     cwd = cwd,
                        #     run = p.cmdline()[1],
                        #     cycle = KepRun(p).cycle(),
                        #     ))
                        # print(p.open_files())
                        # pp = [p.parent]
                        # while pp[-1].pid > 1:
                        #     print('{indent:s}- PARENT: {parent:s}'.format(
                        #         indent = ' '*len(pp),
                        #         parent = pp[-1].name,
                        #         ))
                        #     pp += [pp[-1].parent]
                            # here we have to see whether the next
                            # generation of explosion still uses a shell
                            # if pp.pid > 1:
                            #     ppp = pp.parent
                            #     if ppp.pid > 1 and ppp.name != 'explosion.py':
                            #         print(" -- PARENT: {parent:s}".format(
                            #             parent = ppp.name))

            # if p.status() == psutil.STATUS_ZOMBIE:
            #     print('killing ZOMBIE {:s}'.format(p.name))
            #     p.kill()
            #     p.terminate()
    return jobs

class Fixer(Logged):
    def __init__(self, path = None, run = None):
        assert (path is not None) or (run is not None)
        if path is not None:
            if path.count('/') == 0:
                run, path = path, run
        if run is not None:
            if run.count('/') != 0:
                run, path = path, run
        if path is not None:
            path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        else:
            path = os.path.join(default_path, run)
        if run is None:
            run = os.path.basename(path)
        self.path = path
        self.run = run
        self.dumps = dict()
    def resume(self):
        stop_tokens = ('broken', 'nofix')
        for token in stop_tokens:
            token_filename = os.path.join(self.path, token)
            try:
                os.remove(token_filename)
            except FileNotFoundError:
                pass
        continue_filename = os.path.join(self.path, 'continue')
        touch(continue_filename)
    def get_dump_filename(self, dump):
        return os.path.join(self.path, self.run + dump)
    def get_dump(self, dump = 'z'):
        if not (dump in self.dumps):
            filename = self.get_dump_filename(dump)
            self.dumps[dump] = kepdump._load(filename)
        return self.dumps[dump]
    @contextlib.contextmanager
    def cmdfile(self):
        cmdfile = os.path.join(self.path, self.run + '.cmd')
        with open(cmdfile, 'a+') as f:
            yield f

    def write_cmdfile(self, lines):
        if isinstance(lines, str):
            lines = [lines]
        if lines[-1] != '':
            lines += ['']
        with self.cmdfile() as f:
            f.write('\n'.join(lines))

    def add_cmdfile(self, lines):
        if isinstance(lines, str):
            lines = [lines]
        lines = ['*'] + lines
        self.write_cmdfile(lines)

    def fix(self, force = False):
        nofix_filename = os.path.join(self.path, 'nofix')
        continue_filename = os.path.join(self.path, 'continue')
        with self.logenv(silent = False):
            if os.path.exists(nofix_filename):
                if force:
                    os.remove(nofix_filename)
                else:
                    return
            if os.path.exists(continue_filename):
                return
            self.logger.info('Trying to fix {}'.format(self.run))
            fixers = ['nbkupmax_fixer',
                      'nstop_fixer',
                      'O_Si_knot_fixer',
                      'Si_Fe_knot_fixer',
                      'O_Fe_knot_fixer',
                      'tqsemin_fixer',
                      ]
            for fixer in fixers:
                if getattr(self, fixer)():
                    self.logger.info('Fixed {}'.format(self.run))
                    break
            else:
                self.logger.info('Could not fix {}'.format(self.run))
                touch(nofix_filename)
        # sys.exit()

    @classmethod
    def fix_all(cls,
                path = default_path,
                force = False,
                status = None):
        if status is None:
            status = get_grid_status(path)
        broken = status['broken']
        for b in broken:
            f = cls(os.path.join(path, b), b)
            f.fix(force = force)

    @classmethod
    def restart_all(cls,
                path = default_path,
                status = None):
        if status is None:
            status = get_grid_status(path)
        broken = status['broken']
        broken = ['s33.13']
        for b in broken:
            f = cls(os.path.join(path, b), b)
            f.restart_fixer()

    def restart_fixer(self):
        files = {'log': logdata._load,
                 'cnv': convdata._load,
                 'wnd': winddata._load,
                 }
        with self.chrundir():
            for dump in ('z', 'z1'):
                ok = True
                try:
                    d = self.get_dump(dump)
                    if d is None:
                        ok = False
                except:
                    ok = False

                print('A', ok)

                if not ok:
                    continue
                for f,l in files.items():
                    if not ok:
                        break
                    data = l(self.run + '.' + f, lastmodel = d.qparm.ncyc)
                    print(f,data.nmodels, d.qparm.ncyc)
                    ok &= data.nmodels == d.qparm.ncyc
                if not ok:
                    continue


                print('B', ok)

                d = self.get_dump(dump)
                cmd = []
                cmd += [
                    '@ncyc=={:d}'.format(d.ncyc + step),
                    'k',
                    '']
                self._info('-'*72)
                self._info('Trying {} with ...'.format(d.filename))
                for c in cmd:
                    self._info(c)
                with open('xxx.cmd', 'wt') as f:
                    f.write('\n'.join(cmd))
                args = ['k', 'xxx', d.filename.split(os.path.sep)[-1]]
                with subprocess.Popen(
                    args,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.STDOUT) as proc:
                    message, errmsg = proc.communicate()
                    code = proc.returncode
                self._info('#'*72)
                lines = message.decode().splitlines()
                for line in lines:
                    self._info(line)
                if code == 0 and not lines[-1] == " [TTYCOM] EXECUTE k":
                    self._info('Something is broken.')
                    code = -1
                if code == 0:
                    self.reset_dump(dump)
                    self.resume()
                    self._info('Resuming {:s} with "{:s}"'.format(d.filename, cmd[0]))
                    return True
                else:
                    self._info('Terminated with code {:d}.'.format(code))
            self._info('Could not resume {}.'.format(self.run))
            return False

    def O_Si_knot_fixer(self):
        ok = False
        with np.errstate(invalid='raise'):
            # if there is NaNs in some of the arrays the dump is not
            # good to use; we try to catch this case here.
            for dump in ('z', 'z1'):
                try:
                    d = self.get_dump(dump)
                    if d is None:
                        ok = False
                        continue
                    ok = True
                    ok &= d.netnum[1] == 1
                    ONeMg = (d.net('o16') + d.net('ne20') + d.net('mg24'))[1:-1]
                    ok &= (ONeMg[0]) > 0.9
                    ok &= (d.dn[1] > 5.e7) & (d.tn[1] < 2.e9)
                    ok &= (d.parm.rnmin <= 1)
                    SSi = (d.net('si28') + d.net('s32'))[1:-1]
                    i = np.where(SSi > 0.6)[0]
                    if len(i) > 0:
                        i = i[0] + 1
                    else:
                        i = 0
                    ok &= i > 0
                    if not ok:
                        # fix case of just central ignition this is
                        # particularly to use if p rnmin > 0. and p
                        # 444 > 1 and p o16lim = 0.04
                        ok = True
                        # we have not reduced the parameter yet
                        # (default = 0.04)
                        ok &= d.parm.o16lim > 0.039
                        ok &= (d.tn[1] > 2.e9) & (d.dn[1] > 5.e7)
                        # only trace of O left ...
                        ok &= d.net('o16')[1] < 0.2
                        # ... and only in central zones (compared to
                        # inner 0.1 M_sun) so that it is only 0.1 of
                        # the central value on average
                        i = np.where(d.zm_sun > 0.1)[0][0]
                        ok &= (np.sum(d.net('o16')[1:i] * d.xm[1:i])
                               < 0.1 * d.net('o16')[1] * np.sum(d.xm[1:i]))
                        # intermeditae mass elements make up the rest
                        ok &= (+ d.net('si28')[1]
                               + d.net('s32')[1]
                               + d.net('ca40')[1]
                               + d.net('ar36')[1]) > 0.6
                except FloatingPointError:
                    self.logger.info('Error in ' + d.filename)
                    continue
                except RuntimeWarning:
                    self.logger.info('Dump may have issues: ' + d.filename)
                    continue
                else:
                    break
        if ok:
            lines = [
                'c O_Si_knot_fixer',
                'p rnmin {:8.2e}'.format(d.rn[i+1] * 1.5),
                'p o16lim .005',
                '@tn(1)>{:8.2e}'.format(3.e9),
                'p rnmin -1.']
            self.add_cmdfile(lines)
            self._info('O_Si_knot_fixer')
            self.reset_dump(dump)
            self.resume()
        return ok

    def Si_Fe_knot_fixer(self):
        with np.errstate(invalid='raise'):
            for dump in ('z', 'z1'):
                try:
                    d = self.get_dump(dump)
                    if d is None:
                        ok = False
                        continue
                    ok = True
                    ok &= (d.net('si28')[1] + d.net('s32')[1]) > 0.8
                    iron = d.net.iron()
                    i = np.where(iron[1:-1] > 0.8)[0]
                    if len(i) > 0:
                        i = i[0] + 1
                    else:
                        i = 0
                    ok &= i > 0
                    net = d.netnum
                    ok &= net[1] == 2
                    j = np.where(net[1:-1] == 3)[0]
                    if len(j) > 0:
                        j = j[0] + 1
                    else:
                        j = 0
                    ok &= j == i
                except FloatingPointError:
                    d0 = d
                    self.logger.info('Error in ' + d.filename)
                    pass
                else:
                    break
        if ok:
            lines = ['cpzone {:d} 1 {:d} ltg'.format(j, j-1)]
            self.add_cmdfile(lines)
            self.logger.info('Si_Fe_knot_fixer')
            self.reset_dump(dump)
            self.resume()
        return ok

    def O_Fe_knot_fixer(self):
        with np.errstate(invalid='raise'):
            for dump in ('z', 'z1'):
                try:
                    d = self.get_dump(dump)
                    if d is None:
                        ok = False
                        continue
                    ok = True
                    ok &= (d.net('o16')[1] + d.net('ne20')[1] + d.net('mg24')[1]) > 0.9
                    ok &= (d.dn[1] > 5.e7) & (d.tn[1] < 2.e9)
                    iron = d.net.iron()
                    i = np.where(iron[1:-1] > 0.8)[0]
                    if len(i) > 0:
                        i = i[0] + 1
                    else:
                        i = 0
                    ok &= i > 0
                    net = d.netnum
                    ok &= net[1] == 1
                    j = np.where(net[1:-1] == 3)[0]
                    if len(j) > 0:
                        j = j[0] + 1
                    else:
                        j = 0
                    ok &= d.zm_sun[j] > 0.01
                    ok &= j == i
                except FloatingPointError:
                    d0 = d
                    self.logger.info('Error in ' + d.filename)
                    pass
                else:
                    break
        if ok:
            lines = ['cpzone {:d} 1 {:d} ltg'.format(j, j-1)]
            self.add_cmdfile(lines)
            self.logger.info('O_Fe_knot_fixer')
            self.reset_dump(dump)
            self.resume()
        return ok

    def get_last_output(self):
        base = os.path.join(self.path,self.run)
        pattern = re.compile(base + '_[0-9]+$')
        matches = []
        for f in glob.iglob(base + '_*'):
            matches += pattern.findall(f)
        # matches = [m for m in pattern.findall(f) for f in glob.iglob(base + '_*')]
        tx = 0
        match = None
        for m in matches:
            t = os.stat(m).st_mtime
            if t > tx:
                tx = t
                match = m
        return match

    def nbkupmax_fixer(self,
                       maxbkup = 20,
                       force = False,
                       delta = 10):
        with self.logenv(silent = False):
            lastoutput = self.get_last_output()
            if lastoutput is not None:
                with open(lastoutput, 'rb') as f:
                    f.seek(-1024, os.SEEK_END)
                    line = f.read(1024).decode().splitlines()[-1]
                if line.startswith(" [ZBCYCLE] excess negative abundance backups in zone "):
                    self._info(line)
                    zone = int(line.strip().split()[-4])
                    cycle = int(line.strip().split()[-1])
                    self._info('Died in cycle {cycle:d}, zone {zone:d}'.format(
                        cycle = cycle,
                        zone = zone))
                else:
                    if not force:
                        return False
                    self._info('May not be ZBCYCLE error.')
                d = self.get_dump()
                if d.parm.nbkupmax >= maxbkup and not force:
                    return False
            nbkupmax = d.parm.nbkupmax
            self._info('Old nbkupmax: {:d}'.format(nbkupmax))
            nbkupmax += delta
            self._info('New nbkupmax: {:d}'.format(nbkupmax))
            cmd = ['p nbkupmax {:d}'.format(nbkupmax)]
            self.add_cmdfile(cmd)
            self.reset_dump(d)
            self.resume()
            return True


    @contextlib.contextmanager
    def chrundir(self):
        cwd = os.getcwd()
        os.chdir(self.path)
        yield
        os.chdir(cwd)

    def reset_dump(self, dump):
        if isinstance(dump, kepdump.KepDump):
            f = dump.filename
        else:
            f = self.get_dump_filename(dump)
        if f.endswith('z'):
            return
        f0 = self.get_dump_filename('z')
        self._info('Resuming from ' + f)
        os.remove(f0)
        os.rename(f, f0)

    def tqsemin_fixer(self, plot = False, force = False):
        with self.logenv(silent = False):
            lastoutput = self.get_last_output()
            if lastoutput is not None:
                with open(lastoutput, 'rb') as f:
                    f.seek(-1024, os.SEEK_END)
                    line = f.read(1024).decode().splitlines()[-1]
                if line.startswith(" [SDOTQ] ise can't converge in new zone "):
                    self._info(line)
                    zone = int(line.strip().split(' ')[-1])
                    self._info('Died in zone {:d}'.format(zone))
                else:
                    self._info('May not be TQSE error.')
                    if not force:
                        return False
            d = self.get_dump()
            if d.parm.tqsemin > 2.499e9:
                return
            tqsemin = [2.1e9, 2.1e9, 2.3e9, 2.3e9, 2.5e9, 2.5e9]
            steps =   [   10,    20,    10,    20,    10,    20]
            dumps =   [  'z',  'z1',   'z',  'z1',   'z',  'z1']
            if d.parm.tqsemin > 2.099e9:
                tqsemin = tqsemin[2:]
                steps =     steps[2:]
                dumps =     dumps[2:]
            if d.parm.tqsemin > 2.299e9:
                tqsemin = tqsemin[2:]
                steps =     steps[2:]
                dumps =     dumps[2:]
            for tq, step, dump in zip(tqsemin, steps, dumps):
                with np.errstate(invalid='raise'), self.chrundir():
                    d = self.get_dump(dump)
                    cmd = ['p tqsemin {:8.2e}'.format(tq)]
                    if plot:
                        cmd += ['p 132 1', 'plot']
                    cmd += [
                        '@ncyc=={:d}'.format(d.ncyc + step),
                        'k',
                        '']
                    self._info('-'*72)
                    self._info('Trying {} with ...'.format(d.filename))
                    for c in cmd:
                        self._info(c)
                    with open('xxx.cmd', 'wt') as f:
                        f.write('\n'.join(cmd))
                    args = ['k', 'xxx', d.filename.split(os.path.sep)[-1]]
                    with subprocess.Popen(
                        args,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT) as proc:
                        message, errmsg = proc.communicate()
                        code = proc.returncode
                    self._info('#'*72)
                    lines = message.decode().splitlines()
                    for line in lines:
                        self._info(line)
                    if lines[-1].startswith(" [SDOTQ] ise can't converge in new zone "):
                        self._info('TQSE still broken')
                        code = 1
                    if code == 0 and not lines[-1] == " [TTYCOM] EXECUTE k":
                        self._info('Something else is still broken.')
                        code = -1
                    if code == 0:
                        self.reset_dump(dump)
                        self.add_cmdfile(cmd[0:1])
                        self.resume()
                        self._info('Resuming {:s} with "{:s}"'.format(d.filename, cmd[0]))
                        return True
                    else:
                        self._info('Terminated with code {:d}.'.format(code))
            self._info('May not be TQSE error.')
            return False

    def nstop_fixer(self,
                    delta = 100000,
                    force = False,
                    maxstep = 200000):
        # TODO - add code to backup/reset cnv files etc.
        # but we should not have to do this, really
        d = self.get_dump('#nstop')
        if d is None:
            return False
        if d.qparm.ncyc >= maxstep:
            print('Reached maximum nstop value.')
            if not force:
                return False
        # For now - only extended for O-knot cases
        ok = True
        ok &= (d.net('o16')[1] + d.net('ne20')[1] + d.net('mg24')[1]) > 0.9
        ok &= (d.dn[1] > 5.e7) & (d.tn[1] < 2.e9)
        # check for iron layer
        iron = d.net.iron()
        i = np.where(iron[1:-1] > 0.8)[0]
        if len(i) > 0:
            i = i[0] + 1
        else:
            i = 0
        iFe = i
        # check for silicon/sulfur layer
        silicon = d.net('si28')[1:-1] + d.net('s32')[1:-1]
        i = np.where(silicon > 0.6)[0]
        if len(i) > 0:
            i = i[0] + 1
        else:
            i = 0
        iSi = i
        ok &= ( (iFe > 0) or (iSi > 0 ))
        if not ok:
            if force:
                print('Warning - did not fulfil O-knot criteria.')
            else:
                return False

        nstop = d.parm.nstop
        self._info('Old nstop: {:}'.format(nstop))
        nstop += delta
        self._info('New nstop: {:}'.format(nstop))
        cmd = ['p nstop {:d}'.format(nstop)]
        self.add_cmdfile(cmd)
        self.reset_dump(d)
        self.resume()
        return True

def check_presn(path, base):
    # Hmm, we do not want to check all the time, so we want some
    # record that the run is OK.  We may need some run info data base
    # in the run directory, or somewhere in general.
    dump_filename = os.path.join(path, base + '#presn')
    if not os.path.exists(dump_filename):
        return False
    invalid_tokens = ('broken', )
    for token in invalid_tokens:
        broken_filename = os.path.join(path, token)
        if os.path.exists(broken_filename):
            return False
    presnok_filename = os.path.join(path, 'presn')
    if not os.path.exists(presnok_filename):
        # sometimes file is not quite written to disk, then wait.
        for i in range(10):
            try:
                d = kepdump._load(dump_filename)
            except RecLenError:
                time.sleep(i + 1)
                continue
            else:
                break
        ok = d.is_presn
        if not ok:
            touch(broken_filename)
            print('broken: ', dump_filename)
        else:
            touch(presnok_filename)
            print('ok: ', dump_filename)
        return ok
    return True

def check_all_presn(path = default_path,
                    fix = False,
                    ):
    paths = glob.glob(os.path.join(path, '*'))
    for x in paths:
        print('.', end = '', flush = True)
        run = os.path.basename(x)
        dump_filename = os.path.join(x, run + '#presn')
        presn_filename = os.path.join(x, 'presn')
        broken_filename = os.path.join(x, 'broken')
        if os.path.exists(dump_filename):
            b = os.path.exists(broken_filename)
            p = os.path.exists(presn_filename)
            d = kepdump._load(dump_filename)
            if d.is_presn:
                if b:
                    print('presn seems OK, but marked broken: {}'.format(run))
                    if fix:
                        os.remove(broken_filename)
                if not p:
                    print('presn seems OK, but not marked presn: {}'.format(run))
                    if fix:
                        touch(presn_filename)
            else:
                if p:
                    print('presn not OK, but marked presn: {}'.format(run))
                    if fix:
                        os.remove(presn_filename)
                        os.remove(dump_filename)
                        touch(broken_filename)
        else:
            b = os.path.exists(broken_filename)
            p = os.path.exists(presn_filename)
            if p:
                print('presn not present, but marked presn: {}'.format(run))
                if fix:
                    os.remove(presn_filename)
                    touch(broken_filename)




def clean_broken_presn(path = default_path,
                   fix = False,
                   paths = None,
                   ):
    if paths is None:
        paths = glob.glob(os.path.join(path, '*'))
    for x in paths:
        run = os.path.basename(x)
        dump_filename = os.path.join(x, run + '#presn')
        broken_filename = os.path.join(x, 'broken')
        if (os.path.exists(broken_filename) and
            os.path.exists(dump_filename)):
            d = kepdump._load(dump_filename)
            if d.is_presn:
                print('presn seems OK, but marked proken')
                if fix:
                    os.remove(broken_filename)
                continue
            if fix:
                print('removing {}'.format(dump_filename))
                os.remove(dump_filename)
            else:
                print('found broken {}'.format(dump_filename))


def check_aboarded(path = default_path,
                   filename = None,
                   fix = False,
                   paths = None,
                   ):
    if filename is None:
        timestamp = 0
    else:
        timestamp = os.path.getmtime(filename)
    dirs = dict()
    if paths is None:
        paths = glob.glob(os.path.join(path, '*'))
    for x in paths:
        run = os.path.basename(x)
        # filter out runs just set up
        if not (os.path.exists(os.path.join(x, run + '.cnv')) or
                os.path.exists(os.path.join(x, run + '.wnd')) or
                os.path.exists(os.path.join(x, run + '.log')) or
                os.path.exists(os.path.join(x, run + 'z')) or
                os.path.exists(os.path.join(x, run + 'z1'))):
            continue
        # filter out runs with online status
        if (os.path.exists(os.path.join(x, 'presn')) or
            os.path.exists(os.path.join(x, 'continue')) or
            os.path.exists(os.path.join(x, 'broken')) or
            os.path.exists(os.path.join(x, 'nofix'))):
            continue
        if os.path.getmtime(x) < timestamp:
            continue
        dirs[x] = dict()
    #if fix is not True:
    #    return dirs
    # find last OK dumps
    logs = {
        'cnv': convdata._load,
        'wnd': winddata._load,
        'log': logdata._load,
            }
    for x, v in dirs.items():
        run = os.path.basename(x)
        backup = os.path.join(os.path.dirname(x), 'backup', run)
        if not os.path.exists(backup):
            print('copying directory {} --> {}'.format(x, backup))
            if fix:
                shutil.copytree(x, backup)
        print('checking {} ... '.format(run), end = '', flush = True)
        dumps = [os.path.join(x, run + 'z'),
                 os.path.join(x, run + 'z1')]
        for f in glob.glob(os.path.join(x, run + '#*')):
            dumps += [f]
        dumps_broken = []
        dumps_ok = []
        for f in dumps:
            if not os.path.exists(f):
                continue
            try:
                d = kepdump._load(f)
            except:
                dumps_broken += [f]
            else:
                dumps_ok += [(d.ncyc, f)]
        print(len(dumps_broken), len(dumps_ok))
        dumps_ok = sorted(dumps_ok, key = lambda x: x[0])
        v['dumps_ok'] = dumps_ok
        v['dumps_broken'] = dumps_broken
        for d in dumps_broken:
            print('Removing: ', d, os.path.getsize(d))
            if fix:
                os.remove(d)
        print('last OK dump ', dumps_ok[-1])

        # find last OK history file
        for ext, loader in logs.items():
            data = loader(os.path.join(x, run + '.' + ext),
                          silent = 40,
                          raise_exceptions = False)
            u = data.ncyc
            jj, = np.where(np.not_equal(u[1:], u[:-1]+1))
            if len(jj) == 0:
                ncyc = u[-1]
                print(ext, ncyc)
            else:
                ncyc = u[jj[0]]
                print(ext, 'CORRUPT', ncyc, u[-1], '{:5.2f}%'.format(100* ncyc/ u[-1]))
            v[ext] = ncyc

        max_seq = min([v[ext] for ext in logs.keys()])
        restart_file = None
        for ncyc, f in dumps_ok[::-1]:
            if ncyc <= max_seq:
                restart_file = f
                break
            else:
                print('Removing: ', ncyc, f)
                if fix:
                    os.remove(f)
        print('*'*72)
        print('*** Last complete model: {:d} {}'.format(ncyc, restart_file))
        print('*'*72)

        z_file = os.path.join(x, run + 'z')
        if restart_file != z_file:
            try:
                print('Removing {}'.format(z_file))
                if fix:
                    os.remove(z_file)
            except FileNotFoundError:
                pass
            print('Copying {} --> {}'.format(restart_file, z_file))
            if fix:
                shutil.copy2(restart_file, z_file)
        if fix:
            touch(os.path.join(x, 'continue'), verbose = True)

class GridStatus(Logged):
    def __init__(self,
                 path = default_path,
                 silent = False,
                 ):
        self.path = path
        self.silent = silent
    def __call__(self,
                 path = None,
                 silent = None):
        if path is None:
            path = self.path
        if silent is None:
            silent = self.silent
        with self.logenv(silent = silent):
            runs = find_runs(path)
            current = [KepRun(r).run for r in running()]

            status = OrderedDict()
            status['presn'] = []
            status['online'] = []
            status['broken'] = []
            status['generated'] = []
            status['continued'] = []

            for r in runs:
                if check_presn(os.path.join(path, r), r) is True:
                    status['presn'] += [r]
                elif r in current:
                    status['online'] += [r]
                elif not os.path.exists(os.path.join(path, r, r + 'z')):
                    # more extended tests
                    status['generated'] += [r]
                elif os.path.exists(os.path.join(path, r, 'continue')):
                    status['continued'] += [r]
                else:
                    status['broken'] += [r]

            self.logger.info('running: {:d} {}'.format(len(status['online']), status['online']))
            self.logger.info('generated: {:d} {}'.format(len(status['generated']), status['generated']))
            self.logger.info('continued: {:d} {}'.format(len(status['continued']), status['continued']))
            self.logger.info('broken: {:d} {}'.format(len(status['broken']), status['broken']))
            self.logger.info('presn: {:d}'.format(len(status['presn'])))

        return status

get_grid_status = GridStatus()


def status(path = default_path):
    current = []
    for r in running():
        try:
            current += [KepRun(process = r)]
        except:
            print('failed for', r)
            raise
    for r in current:
        p = r.process
        t = p.cpu_times()
        rt = time.time() - p.create_time()
        ut = t.user + t.system
        print(r.cwd,
              r.run,
              r.cycle(),
              time2human(rt),
              time2human(ut),
              '{:5.1f}%'.format(ut / rt * 100),
              )

class Batch(Logged):
    # should really get a list of setup object or read config file
    def __init__(self,
                 path = default_path,
                 cores = None):
        if cores is None:
            cores = psutil.cpu_count()
        self.path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        self.cores = cores

    def start(self, run, mode = 'new', nice = 19):
        self.setup_logger(False)

        # this should become a method of a run object
        path = os.path.join(self.path, run)
        kepler = 'k'

        args = [kepler, run]
        if mode == 'continue':
            args += ['z']
            continuefile = os.path.join(path, 'continue')
            if os.path.exists(continuefile):
                self.logger.info('removing ' + continuefile)
                os.unlink(continuefile)
        elif mode == 'new':
            args += ['g']
        else:
            raise Exception('Wrong mode')

        p = psutil.Popen(
            args,
            shell  = False,
            cwd    = path,
            stdin  = subprocess.DEVNULL,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            start_new_session = True)
        if (nice is not None) and (nice > 0):
            p.nice(nice)
        self.logger.info(
            'Starting {run:s} with PID {pid:d}'.format(
                run = ' '.join(p.cmdline()),
                pid = p.pid))
          # self.p = p

        # delete tokens
        bad_tokens = ('nofix', 'broken', 'presn')
        for token in bad_tokens:
            token_filename = os.path.join(path, token)
            try:
                os.unlink(token_filename)
                self.logger.info('removing ' + continuefile)
            except FileNotFoundError:
                pass

        self.close_logger()


    def run(self, sleep = 60, cores = None):
        first = True
        if cores is None:
            cores = self.cores
        i = 0
        while True:
            # this should all be replaced by unified KepRun objects

            if first:
                first = False
            else:
                time.sleep(sleep)
            status = get_grid_status(self.path)
            free = cores - max(len(status['online']), int(os.getloadavg()[0]))

            self.setup_logger(False)
            if free <= 0:
                self.logger.info('Nothing to do.')
            if free > 0:

                # sort generated files by time
                # we should even process continuations by age of generators ...

                continued = status['continued']

                times = []
                for r in continued:
                    genfile = os.path.join(self.path, r, r + 'g')
                    timeg = os.path.getmtime(genfile)
                    times += [timeg]
                ii = np.argsort(times)
                continued = np.array(continued)[ii].tolist()

                for r in continued:
                    continuefile = os.path.join(self.path, r, 'continue')
                    if os.path.exists(continuefile):
                        self.start(r, mode = 'continue', nice = 10)
                        free -= 1
                        if free <= 0:
                            break
            if free > 0:

                generated = status['generated']

                times = []
                for r in generated:
                    genfile = os.path.join(self.path, r, r + 'g')
                    timeg = os.path.getmtime(genfile)
                    times += [timeg]
                ii = np.argsort(times)
                generated = np.array(generated)[ii].tolist()

                for r in generated:
                    self.start(r, mode = 'new')
                    free -= 1
                    if free <= 0:
                        break

            # fix broken runs
            Fixer.fix_all(
                path = self.path,
                status = status,
                )

            # clean directories
            if i % 60 == 0:
                self._info('Cleaning directories')
                for r in status['presn']:
                    cwd = os.path.join(self.path, r)
                    if not is_clean(cwd, r):
                        KepRun(cwd = cwd, run = r).clean()
                self._info('Done cleaning directories')

            # add features for packaging
            # likely using derived class

            self.close_logger()
            i += 1


if __name__ == "__main__":
    # here we should use argument processing
    # check behaviour based on program name

    vi = sys.version_info
    assert (vi.major >= 3 and vi.minor >= 4) or vi.major > 3

    argv = sys.argv
    args = argv[1:]
    kwargs = dict()
    for i,a in enumerate(args):
        try:
            args[i] = eval(a)
        except:
            pass
    for a in list(args):
        try:
            d = eval('dict('+a+')')
            kwargs.update(d)
            args.remove(a)
        except:
            pass

    b=Batch(*args, **kwargs)
    b.run()

    status()
    logging.shutdown()
