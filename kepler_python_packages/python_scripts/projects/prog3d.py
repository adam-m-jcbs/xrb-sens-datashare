"""
Project for 3D progenitor models
"""

import shutil
import os
import os.path
import glob
import tarfile
import re
import psutil
import subprocess

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

import numpy as np

import kepgen, kepdump, winddata, kepdata, convdata, convplot
import color
import physconst

from collections import Iterable

from utils import iterable, is_iterable
from human import byte2human
from logged import Logged
from batch import compress_file

project = 'prog3d'
composition = 'solas12'
sentinel = 's'

modes = {
    'detailed' : {'commands' : '\n'.join(
        ('//',
         '@tn(1)>3.d9',
         'd #siign',
         'p nsdump 1',
         'p ndump 1',
         )),
        },
    'normal' : {'commands' : '\n'.join(
        ('//',
         '@tn(1)>3.d9',
         'd #siign',
         )),
        },
}

runpar = dict(
    composition = composition,
    #  mass=15,
    dirtarget = os.path.join(project, sentinel + '{mass}'),
    burn = True,
    bdat = 'rath00_10.1.bdat_jlf_mal-hf181',
    extra = 'p 52 20',
    # yeburn = True
    )

class Presn(Logged):
    def __init__(self,
                 load = None,
                 keep = None,
                 burn = False,
                 dump = 'presn',
                 rundir = None,
                 template = None,
                 ):
        """Default is to load and keep files

        If keep is set to True but load is not set, the default is not
        to load.
        """
        if load is None:
            if keep is None:
                load = True
            else:
                load = False
        if load is True:
            keep = True
        elif keep is None:
            keep = load
        self.keep = keep
        self.burn = burn
        self.dump = dump
        self.template = template

        self.ke = kepgen.KepEnv(silent = True)
        if rundir == None:
            self.rundir = os.path.join(self.ke.dir00, project)
        else:
            self.rundir = os.path.expandvars(os.path.expanduser(rundir))
        files = self._get_files()
        print('Loading {:d} models.'.format(len(files)))
        if load:
            self.dumps = {f: self._loaddump(f) for f in files}
        else:
            self.dumps = {f: None for f in files}
        self.names = None

    def _loaddump(self, filename):
        return kepdump.load(filename, killburn = not self.burn)

    def _get_files(self):
        if self.template == None:
            template = os.path.join(sentinel+'*', sentinel+'*')
        else:
            template = self.template
        return sorted(glob.glob(os.path.join(self.rundir, template+'#'+self.dump)))

    def update(self, load = True):
        files_updated = set(self._get_files())
        files_old = set(self.dumps)
        if self.keep:
            dumps = {}
            for f in files_updated:
                d = self.dumps.pop(f, None)
                if d is None and load:
                    d = self._loaddump(f)
                dumps[f] = d
            self.dumps = dumps
        else:
            self.dumps = {f: None for f in files_updated}
        removed = len(files_old - files_updated)
        added = len(files_updated - files_old)
        if added > 0:
            print('Number of files added: {:d}'.format(added))
        if removed > 0:
            print('Number of files removed: {:d}'.format(removed))
        if added == removed == 0:
            print('Up to date.')
        else:
            self.names = None

    def __iter__(self):
        yield from self.iterdumps()

    def clear_dumps(self):
        self.dums = {f: None for f in self.dumps.keys()}

    def iterdumps(self, keep = None):
        if keep is None:
            keep = self.keep
        for f,d in self.dumps.items():
            if d is None:
                d = self._loaddump(f)
                if keep:
                    self.dumps[f] = d
            yield d
    def itermass(self):
        for d in self.iterdumps():
            yield d.mass_string
    def iterfiles(self):
        yield from self.dumps.keys()
    def __len__(self):
        return len(self.dumps)
    def __getitem__(self, index, keep = None):
        if keep is None:
            keep = self.keep
        if self.names is None:
            self.names = sorted(list(self.iterfiles()))
        d = self.dumps[self.names[index]]
        if d is None:
            d = self._loaddump(f)
            if keep:
                self.dumps[f] = d
        return d
    def __delitem__(self, index):
        if self.names is None:
            self.names = sorted(list(self.iterfiles()))
        del self.dumps[self.names[index]]
        del self.names[index]
    def __setitem__(self, index, value):
        if self.names is None:
            self.names = sorted(list(self.iterfiles()))
        self.dumps[self.names[index]] = value

masses = np.sort(np.hstack((
    np.linspace(17.5, 21, 351),
)))

def gen(**parm):
    """
    Make run

    parameter used:

     - mass:
         array of masses

     - run:  True | False
         whether to start run immediately

     - mode: 'normal'
         select cases
           'normal' - just write #siign
           'detailed' - write all dumps from #siign

    - after_file:
         file to use as basis for next dates

    """
    mass = parm['mass']
    if is_iterable(mass):
        parm = dict(parm)
        after_file = parm.pop('after_file', None)
        if after_file is not None:
            xtime = os.path.getmtime(after_file)
        for m in mass:
            p = dict(parm)
            if after_file is not None:
                xtime += 1
                p['genfiletime'] = xtime
            p['mass'] = m
            gen(**p)
        return

    p = dict(runpar)
    mode = parm.pop('mode', 'normal')
    p.update(modes[mode])

    # p.update(specials[case])
    p.update(parm)
    p['run'] = parm.pop('run', True)
    form = p['dirtarget']
    p['dirtarget'] = form.format(mass = kepgen.mass_string(mass))

    after_file = parm.get('after_file', None)
    if after_file is not None:
        xtime = os.path.getmtime(after_file) + 1
        p['genfiletime'] = xtime

    kepgen.MakeRun(**p)


def mass2str(parm):
    """
    convert mass to mass tring in dictionary
    """
    mass = parm['mass']
    if not isinstance(mass, str):
        parm = dict(parm)
        parm['mass'] = kepgen.mass_string(mass)
    return parm

def find_model(path = None, run = None, time = None):
    if path is None:
        path = os.getcwd()
    assert run is not None
    assert time is not None
    presn = kepdump.load(os.path.join(path, run + '#presn'))
    dt = []
    ff = []
    for f in glob.glob(os.path.join(path, run + '#[0-9]*')):
        d = kepdump.load(f)
        dt += [presn.parm.time - d.parm.time - time]
        ff += [f]
        print('.', end = '', flush = True)
    dt = np.array(dt)
    i = np.argmin(dt**2)
    print('\n', ff[i])


def find_model(base = None, time = 300, offset = 0.25):
    c = convdata.load(base + '.cnv')
    tcc = c.timecc(offset)
    ii = np.argmin(np.abs(tcc - time))
    print('Model: {}, time = {}'.format(
        c.models[ii],
        c.time[-1] - c.time[ii] + offset))

def find_models(**parm):
    """
    1) about 1/2 hour before core collapse
    2) about 1 hour before core collapse
    3) about 3 hours before collapse
    4) about 1 day before collapse
    5) at the onset of core Si burning
    """
    rundir = parm.pop('rundir', None)
    mass2str(parm)
    mass = parm['mass']
    dirtarget = runpar['dirtarget'].format(**mass2str(parm))
    ke = kepgen.KepEnv(silent = True)
    if rundir is None:
        rundir = os.path.join(ke.dir00, dirtarget)
    run = 's{mass}'.format(mass = kepgen.mass_string(mass))
    presn = kepdump.load(os.path.join(rundir, run + '#presn'))
    sidep = kepdump.load(os.path.join(rundir, run + '#sidep'))
    siign = kepdump.load(os.path.join(rundir, run + '#siign'))

    times = [presn.parm.time - siign.parm.time - 1, 24*3600, 3*3600, 3600, 1800, (presn.parm.time - sidep.parm.time) - 1e-6, 0]
    names = ['siign', 't-86400', 't-10800', 't-3600', 't-1800', 'sidep', 'presn']

    # shortcut to times - just load wind data
    wind = winddata.load(os.path.join(rundir, run + '.wnd'))
    cycles = np.arange(siign.qparm.ncyc, presn.qparm.ncyc+1)
    tcc = wind.tcc(0)[cycles - 1]

    models = []
    for t in times:
        i = np.argwhere(t >= tcc)[0][0]
        models += [cycles[i]]
        cycles = cycles[i+1:]
        tcc = tcc[i+1:]

    return models, names

def extract_mr_history(
        path = None,
        base = None,
        start = '#6min',
        end = '#presn',
        m = None,
        r0 = None,
        ):
    if path is None:
        path = os.getcwd()
    if base is None:
        files = glob.glob(os.path.join(path, '*#presn*'))
        bases = [re.findall('/([^/]+)#presn(?:\.(?:xz|gz|bz))?$', f)[0]
                for f in files]
        if len(bases) == 1:
            base = bases[0]
        else:
            for b in bases:
                if b not in ('xxx',):
                    base = b
                    break
        print(f' [extract_mr_history] Using base {base}')
    assert isinstance(base, str)
    d0 = kepdump.load(os.path.join(path, base + start))
    d1 = kepdump.load(os.path.join(path, base + end))
    if r0 is not None:
        i = np.where(d0.rn >= r0)[0][0] - 1
        m = d0.zm_sun[i] + 4 * np.pi / 3 * (r0**3 - d0.rn[i]**3) * d0.dn[i+1] / physconst.Kepler.solmass
    assert m is not None, 'need to specify m or r0'
    with open(os.path.join(path, base + f'_m{m:04.2f}_{start.replace("#","")}.txt'), 'tw') as f:
        for cycle in range(d0.qparm.ncyc, d1.qparm.ncyc + 1):
            print('.', end = '', flush = True)
            d = kepdump.load(os.path.join(path, base + '#{:d}'.format(cycle)))
            r = d.rn[:-1]
            i = np.where(d.zm_sun >= m)[0][0] - 1
            t = (m - d.zm_sun[i]) / d.xm_sun[i + 1]
            r = ((1 - t) * r[i]**3 + t * r[i + 1]**3)**(1/3)
            u = (1 - t) * d.un[i] + t * d.un[i + 1]
            time = d.parm.time - d0.parm.time
            if d.parm.toffset != d0.parm.toffset:
                time += d.parm.toffset - d0.parm.toffset
            f.write(f'{time:25.17e} {r:25.17e} {u:25.17e}\n')


def make_named_dumps(**parm):
    mass = parm['mass']
    if isinstance(mass, (Iterable, np.ndarray)) and not isinstance(mass, str):
        for m in mass:
            p = dict(parm)
            p['mass'] = m
            make_named_dumps(**p)
        return

    models, names = find_models(**parm)
    mass = parm['mass']
    dirtarget = runpar['dirtarget'].format(**mass2str(parm))
    ke = kepgen.KepEnv(silent = True)
    rundir = os.path.join(ke.dir00, dirtarget)
    run = 's{mass}'.format(**mass2str(parm))
    for m,n in zip(models, names):
        numbered = os.path.join(rundir, run + '#{:d}'.format(m))
        named = os.path.join(rundir, run + '#{:s}'.format(n))
        if os.path.exists(named):
            print('{} already exists - not overwriting.'.format(named))
        else:
            shutil.copy2(numbered, named)

def txt_data(**parm):
    models, names = find_models(**parm)
    # if parm.get('names', False) == True:
    #     models = names
    mass = parm['mass']
    dirtarget = runpar['dirtarget'].format(**mass2str(parm))
    ke = kepgen.KepEnv(silent = True)
    rundir = os.path.join(ke.dir00, dirtarget)
    run = 's{mass}'.format(**mass2str(parm))
    for m,n in zip(models, names):
        dumpfile = os.path.join(rundir, run + '#' + n)
        if not os.path.exists(dumpfile):
            dumpfile = os.path.join(rundir, run + '#{:d}'.format(m))
            copy = True
        else:
            copy = False
        if not os.path.exists(dumpfile):
            print('{} does not exists.'.format(dumpfile))
            continue
        dump = kepdump.load(dumpfile)
        outfile = os.path.join(rundir, run + '@' + n)
        kepdata.KepData(
            dump = dump,
            burn = False,
            outfile = outfile,
        )
        outfile = os.path.join(rundir, run + '@@' + n)
        kepdata.KepData(
            dump = dump,
            burn = True,
            outfile = outfile,
        )
        if parm.get('copy', copy) == True:
            named = os.path.join(rundir, run + '#{:s}'.format(n))
            if os.path.exists(named):
                print('{} already exists - not overwriting.'.format(named))
            else:
                shutil.copy2(dumpfile, named)

def package(**parm):
    """
    make files and create package on server

    set make_tar=True to make tar files
    """
    def filter(ti):
        """Filter out *_* files"""
        if ti.name.count('_') > 0:
            return None
        return ti

    mass = parm['mass']
    if isinstance(mass, (Iterable, np.ndarray)) and not isinstance(mass, str):
        for m in mass:
            p = dict(parm)
            p['mass'] = m
            package(**p)
        return

    txt_data(**parm)

    dirtarget = runpar['dirtarget'].format(**mass2str(parm))
    ke = kepgen.KepEnv(silent = True)
    rundir = os.path.join(ke.dir00, dirtarget)
    webdir = os.path.join(os.path.abspath(os.sep), 'm','web','Download', project, runpar['composition'])
    for fn in glob.glob(os.path.join(rundir, '*@*')):
        target = os.path.join(webdir, os.path.split(fn)[-1])
        if os.path.exists(target):
            if parm.get('overwrite', False) == True:
                os.remove(target)
            else:
                s = 'file {} already exists'.format(target)
                print('-'*72)
                print(s)
                print('Use overwrite = True to overwrite files.')
                print('-'*72)
                raise Exception(s)
        shutil.move(fn, webdir)
    for fn in glob.glob(os.path.join(rundir, '*_*')):
        os.remove(fn)
    run = 's{mass}'.format(mass = kepgen.mass_string(mass))
    if parm.get('make_tar', False) == True:
        with tarfile.open(os.path.join(webdir, run + '.tgz'), 'w:gz') as tf:
            tf.add(rundir, filter = filter, arcname=run)



def compactness(xim=2.5, axis = 'mass', dumps = None, masses = None):
    """
    Return compactness parameter

    Returns a tupe of mass and corresponding compactness parameter,
    both as numpy arrays.
    """

    def dump_iter():
        try:
            yield from dumps
        except:
            for mass in masses:
                ms = kepgen.mass_string(mass)
                run = 's{mass}'.format(mass = ms)
                dumpfile = os.path.join(ke.dir00, project, run, run + '#presn')
                dump = kepdump.load(dumpfile)
                yield dump

    if dumps is None:
        ke = kepgen.KepEnv(silent = True)
    c = []
    m = []

    for dump in dump_iter():
        core = dump.core()
        if axis == 'mass':
            m += [dump.mass]
        else:
            m += [core[axis].zm_sun]
        c += [dump.compactness(xim)]

    c = np.array(c)
    m = np.array(m)
    return m, c


def plot_dn():
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel(r'$m \; / \; \mathrm{M}_\odot$')
    ax.set_ylabel(r'$\rho\; /\; \mathrm{g}\,\mathrm{cm}^{-3}$')
    ax.set_xlim(0,4)
    ax.set_ylim(7.e3,4.e9)

    masses = np.linspace(18, 21, 31)
    ke = kepgen.KepEnv(silent = True)
    c = color.isocolors(len(masses))
    for i,mass in enumerate(masses):
        ms = kepgen.mass_string(mass)
        run = 's{mass}'.format(mass = ms)
        dumpfile = os.path.join(ke.dir00, project, run, run + '#presn')
        dump = kepdump.load(dumpfile)
        x = dump.zm_sun
        y = dump.dn
        ax.plot(x, y,
                color = c[i],
                label = r'${} \,\mathrm{{M}}_\odot$'.format(kepgen.mass_string(mass)))
    ax.legend(loc = 'best', ncol=2, fontsize=10)
    f.tight_layout()
    plt.draw()

def clean(**parm):
    mass = parm['mass']
    if isinstance(mass, (Iterable, np.ndarray)) and not isinstance(mass, str):
        for m in mass:
            p = dict(parm)
            p['mass'] = m
            clean(**p)
        return

    ke = kepgen.KepEnv(silent = True)
    ms = kepgen.mass_string(mass)
    run = 's{mass}'.format(mass = ms)
    rundir = os.path.join(ke.dir00, project, run)
    print('cleaning {}'.format(rundir))

    del_list = (
        '^xxx.*',
        '^' + run + '#[0-9]+$',
        '^' + run + 'z[0-9]*$',
        '^' + run + '_[0-9]+$',
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
                print('removing {}'.format(fn))
                os.remove(fn)
                deleted_bytes += stat.st_size
                deleted_files += 1
                break
        else:
            for x in compress_list:
                if len(x.findall(name)) > 0:
                    print('compressing {}'.format(fn))
                    compress_file(fn)
                    compressed_bytes += stat.st_size
                    compressed_files += 1
                    break
            else:
                print('keeping {}'.format(fn))
                kept_bytes += stat.st_size
                kept_files += 1

    print("deleted    {:5d} files ({})\ncompressed {:5d} files ({})\nkept       {:5d} files ({})\n".format(
        deleted_files,
        byte2human(deleted_bytes),
        compressed_files,
        byte2human(compressed_bytes),
        kept_files,
        byte2human(kept_bytes),
    ))

class CorePlot(Logged):
    def __init__(self, dumps = None, **kwargs):
        self.dumps = dumps
        self.f = None
        self.plot(**kwargs)

    def newfigure(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        self.ax = ax
        self.f = f
    #     self.close_event_id = f.canvas.mpl_connect('close_event', self.close_event)

    # def close_event(self, event):
    #     self.f.canvas.mpl_disconnect(self.close_event_id)
    #     self.f = None

    def plot(self, **kwargs):
        try:
            f = self.f
            if f.canvas.manager.window is None:
                raise Exception()
            ax = self.ax
            xlim0 = ax.get_xlim()
            ylim0 = ax.get_ylim()
            ax.clear()
        except:
            self.newfigure()
            f = self.f
            ax = self.ax
            xlim0 = None
            ylim0 = None

        dumps = self.dumps
        if dumps is None:
            dumps = Presn()
            self.dumps = dumps

        co = []
        he = []
        ne = []
        si = []
        os = []
        fe = []
        st = []
        m = []
        fail = []
        for dump in dumps:
            core = dump.core()
            m += [dump.mass]
            st += [core['star'].zm_sun]
            he += [core['He core'].zm_sun]
            co += [core['C/O core'].zm_sun]
            ne += [core['Ne/Mg/O core'].zm_sun]
            os += [core['O shell'].zm_sun]
            try:
                si += [core['Si core'].zm_sun]
            except:
                fail += [dump.mass]
                si += [0.]
                fe += [0.]
                continue
            try:
                fe += [core['iron core'].zm_sun]
            except:
                fail += [dump.mass]
                fe += [0.]

        print('Failed: ', fail)

        co = np.array(co)
        he = np.array(he)
        ne = np.array(ne)
        si = np.array(si)
        os = np.array(os)
        fe = np.array(fe)
        st = np.array(st)
        m = np.array(m)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        xlabel = r'initial mass / solar masses'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'core mass / solar masses')

        ax.plot(m, he, '+', label = 'helium')
        ax.plot(m, co, '+', label = 'carbon/oxygen')
        ax.plot(m, ne, '+', label = 'neon/oxygen')
        ax.plot(m, si, '+', label = 'silicon/sulphur')
        ax.plot(m, os, '+', label = 'oxygen shell')
        ax.plot(m, fe, '+', label = 'iron core')
        ax.plot(m, st, '+', label = 'star')

        if xlim0 is None:
            xlim0 = None
        xlim = kwargs.get('xlim', xlim0)
        if xlim is None:
            xlim = [min(m)-0.1, max(m)+0.1]

        if ylim0 is None:
            ylim0 = [1.2, 5.8]
        ylim = kwargs.get('ylim', ylim0)
        if ylim is None:
            ylim = [0, max(st)*1.025]

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.legend(loc = 'best', numpoints = 1)
        f.tight_layout()
        f.show()

    def update(self, *args, **kwargs):
        self.dumps.update()
        self.plot(*args, **kwargs)

    def special_log_xaxis(self):
        self.ax.set_xscale('log')
        self.ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        self.ax.set_xticks([10,12,15,20,30,40])
        self.ax.xaxis.set_major_formatter(ScalarFormatter())

        self.ax.set_ylim((1.2,19.99))
        self.ax.set_yscale('log')
        self.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        self.ax.set_yticks([2,5,10,15])
        self.ax.yaxis.set_major_formatter(ScalarFormatter())

        self.f.tight_layout()
        self.f.show()

class SOPlot(Logged):
    def __init__(self, dumps = None, **kwargs):
        self.dumps = dumps
        self.f = None
        self.plot(**kwargs)

    def newfigure(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        self.ax = ax
        self.f = f

    def plot(self, **kwargs):
        try:
            f = self.f
            if f.canvas.manager.window is None:
                raise Exception()
            ax = self.ax
            xlim0 = ax.get_xlim()
            ylim0 = ax.get_ylim()
            ax.clear()
        except:
            self.newfigure()
            f = self.f
            ax = self.ax
            xlim0 = None
            ylim0 = None

        dumps = self.dumps
        if dumps is None:
            dumps = Presn()
            self.dumps = dumps

        so = []
        sn = []
        m = []
        for dump in dumps:
            core = dump.core()
            m += [dump.mass]
            so += [core['O shell'].stot]
            sn += [np.log10(dump.sn[core['O shell'].j])]

        so = np.array(so)
        sn = np.array(sn)
        m = np.array(m)

        # f = plt.figure()
        # ax = f.add_subplot(111)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_xlabel(r'initial mass / solar masses')

        ax.plot(m, so, 'b+',)
        ax.set_ylabel(r'entropy / $k_\mathrm{B}$ per baryon', color = 'b')

        ax2 = ax.twinx()
        ax2.plot(m, sn, 'g+')
        ax2.set_ylabel(r'log( energy generation / erg/g/sec )', color = 'g')

#    ax.set_ylim([1.2, 5.8])
        ax.set_xlim([min(m)-0.1, max(m)+0.1])
        # ax.legend(loc = 'best', numpoints = 1)
        # ax2.legend(loc = 'best', numpoints = 1)
        f.tight_layout()
        plt.draw()

    def update(self, *args, **kwargs):
        self.dumps.update()
        self.plot(*args, **kwargs)


class CompPlot(Logged):
    """
    Plot compactness parameters.

    best to provide an array of dums in KW "dumps"
    KW axix allows values of
     - 'mass'
     - 'C/O core'
     - 'He core'
    """
    def __init__(self,
                 dumps = None,
                 xi = [1.5,2.0,2.5,3.0],
                 axis = 'mass',
                 ):

        f = plt.figure()
        ax = f.add_subplot(111)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        labels = {'mass' : r'initial mass / solar masses',
                  'He core' : r'He core mass / solar masse',
                  'C/O core' : r'CO core mass / solar masses',
              }
        ax.set_xlabel(labels[axis])
        ax.set_ylabel(r'compactness parameter $\xi_m$')

        if not isinstance(xi, (Iterable, np.ndarray)):
            xi = [xi]
        if dumps is None:
            dumps = Presn()
        ym = []
        for xii in xi:
            xm, xi = compactness(xim = xii, axis = axis, dumps = dumps)
            ax.plot(xm, xi, '+', label = r'$m = {}$'.format(kepgen.mass_string(xii)))
            ym += [max(xi)]
        ax.set_xlim([min(xm)-0.1, max(xm)+0.1])
        ax.set_ylim([0, max(ym)*1.05])
        ax.legend(loc = 'best')
        f.tight_layout()
        plt.draw()
        self.f = f

class ErtlPlot(Logged):
    """
    Plot Ertle SN parameters.

    best to provide an array of dums in KW "dumps"
    KW axix allows values of
     - 'mass'
     - 'C/O core'
     - 'He core'
    """
    def __init__(self,
                 dumps = None,
                 axis = 'mass',
                 ):

        f = plt.figure()
        ax = f.add_subplot(111)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        labels = {'mass' : r'initial mass / solar masses',
                  'He core' : r'He core mass / solar masse',
                  'C/O core' : r'CO core mass / solar masses',
              }
        ax.set_xlabel(labels[axis])
        ax.set_ylabel(r'$M_4 \mu_4$, $\mu_4$ (solar units)')
        # ax.set_ylabel(r'$M_4 $, $1/\mu_4$')

        if dumps is None:
            dumps = Presn()
        m = []
        x = []
        y = []
        for dump in dumps:
            x_, y_ = dump.ertl
            # x_, y_ = dump.ertl2
            m += [dump.mass]
            x += [x_]
            y += [y_]

        ax.plot(m, x, '+', label = r'$M_4\,\mu_4$')
        ax.plot(m, y, 'x', label = r'$\mu_4$')

        # ax.plot(m, x, '+', label = r'$M_4$')
        # ax.plot(m, y, 'x', label = r'$1/\mu_4$')

        ax.set_xlim([min(m)-0.1, max(m)+0.1])
        ax.set_ylim([0, max(max(x),max(y))*1.05])
        ax.legend(loc = 'best')
        f.tight_layout()
        plt.draw()
        self.f = f

def plot_results(dumps = None, cores = 'mass'):
    if dumps is None:
        dumps = Presn()
    print('Plotting {:d} models.'.format(len(dumps)))
    plots = []
    for core in iterable(cores):
        plots += [CompPlot(axis = core, dumps = dumps)]
    return plots, dumps

def kepler_update(mass):
    masses = iterable(mass)
    source = '/home/alex/kepler/gfortran/keplery'
    ke = kepgen.KepEnv(silent = True)
    for mass in masses:
        dirtarget = runpar['dirtarget'].format(mass=kepgen.mass_string(mass))
        rundir = os.path.join(ke.dir00, dirtarget)
        target = os.path.join(rundir, 'k')
        print('copying {} --> {}'.format(source, target))
        try:
            shutil.copy2(source, target)
            continue
        except OSError:
            print('Deleting old {}'.format(target))
            os.remove(target)
        shutil.copy2(source, target)

def excessplot():
    f = plt.figure()
    a = f.add_subplot(111)

    d = kepdump.load('~/kepler/prog3d/s15/s15#cign')
    a.plot(d.zm_sun,d.netb.eta,label='post-He')
    d = kepdump.load('~/kepler/prog3d/s15/s15#odep')
    a.plot(d.zm_sun,d.netb.eta,label='central O-dep')
    d = kepdump.load('~/kepler/prog3d/s15/s15#presn')
    i = np.where(d.zm > d.parm.bmasslow)[0][0]
    a.plot(d.zm_sun[i:],d.netb.eta[i:],label='pre-SN')

    a.set_xlabel('mass / solar masses')
    a.set_ylabel(r'neutron excess $\eta$')

    a.legend(loc='best')

    a.set_xlim(1.2,4)
    a.set_ylim(-0.000, 0.0027)
    f.tight_layout()
    f.show()

def compactness_data(cm = 2.5):
    p = Presn()
    c = np.array([[x.mass,x.compactness(cm)] for x in p])
    with open('/home/alex/Downloads/prog3d_zeta.txt', 'wt') as f:
        for x in c:
            f.write('{:12.5f}  {:12.5f}\n'.format(*x.tolist()))


def c12data(c = None, p = None):
    """
    Extract data for high resolution comparison project with Stan and Tug
    """
    if c is None:
        c = Presn(dump='cign')
    c12 = {x.mass: x.abu.c12[1] for x in c}
    if p is None:
        p = Presn()
    data = []
    for x in p:
        try:
            d = [
                x.mass,
                x.core()['star'].zm_sun,
                x.core()['He core'].zm_sun,
                x.core()['C/O core'].zm_sun,
                x.core()['iron core'].zm_sun,
                c12[x.mass],
                ]
        except KeyError as e:
            print(x.mass, e)
        data.append(d)
    data = sorted(data, key=lambda x: x[0])
    with open('/home/alex/Downloads/prog3d_tug.txt', 'wt') as f:
        for x in data:
            f.write(('{:12.5f} '*len(x) + '\n').format(*x))
    return c, p

# # COPY DATA
# cd
# sshfs c:kepler/prog3d x
# ip
# import prog3d
# import subprocess
# p = prog3d.Presn(rundir = 'x')
# for d in p:
#     if d.mass < 12:
#         name = str(d.mass).rstrip('0').rstrip('.')
#         subprocess.run("rsync -vazu --progress -e ssh c:kepler/prog3d/s{} .".format(name), shell = True, cwd = '/travel1/alex/kepler/prog3d')

def cnvplot(
    path = '/home/alex/kepler/test/'):
    models = {
        # 's12.9' : {'ymax' : 2.8},
        # 's14.4' : {'ymax' : 3.5},
        # 's15.4' : {'ymax' : 4.3},
        's16.3' : {'ymax' : 4.8},
        # 's17.1' : {'ymax' : 5.1},
        # 's17.57' : {'ymax' : 5.3},
        }
    for model, parameters in models.items():
        filename = os.path.join(path, model, model + '.cnv.xz')
        c = convdata.load(filename)
        p = convplot.plot(
            c,
            logtime = -8.2,
            stability = ['conv'],
            mingain=4,
            minloss=4,
            solar=True,
            xlim=[4,-8.2],
            ymax = parameters.get('ymax', None))
        filename = os.path.join(path, model + '.cnv.pdf')
        p.fig.savefig(filename)

def entplot(
    path = '/home/alex/kepler/test/'):
    models = {
        's12.9' : {},
        's14.4' : {},
        's15.4' : {},
        's16.3' : {},
        's17.1' : {},
        's17.57' : {},
        }
    f = plt.figure()
    ax = f.add_subplot(111)
    for model, parameters in models.items():
        filename = os.path.join(path, model, model + '#presn')
        d = kepdump.load(filename)
        m = d.zmm_sun[1:]
        ax.plot(m, d.stot[1:], label = r'$' + model[1:] + r'\,\mathrm{M}_\odot$')
    ax.set_xlim(1.3,3.7)
    ax.set_ylim(3.8,5.9)
    ax.set_xlabel('enclosed mass (solar masses)')
    ax.set_ylabel(r'specific entropy ($k_\mathrm{B}$ / baryon)')
    plt.tight_layout()
    ax.legend(loc='best')
    f.savefig(os.path.join(path, 'S.pdf'))


#######################################################################

# prog3d.gen(mass=np.linspace(20.1, 21, 10), mode = 'detailed')
# prog3d.package(mass=np.linspace(20.1, 21, 10))
# prog3d.plot_comp([1.5,2,2.5,3])
# prog3d.plot_dn([1.5,2,2.5,3])
# prog3d.package(mass=np.linspace(18, 19.9, 20), make_tar = True, overwrite=True)
# prog3d.gen(mass=linspace(18.81, 18.89, 9), mode = 'detailed')
# prog3d.gen(mass=[18.73, 18.74, 18.75, 18.76, 19.01, 19.06, 19.02, 19.07, 19.11, 19.09, 19.08, 19.12, 19.15, 19.16, 19.19, 19.04],mode = 'detailed')
# prog3d.package(mass=[19.17, 19.14, 18.79, 18.71, 18.72, 19.18, 18.78, 19.03, 19.05, 18.77, 19.13])
# prog3d.clean(mass=[19.17, 19.14, 18.79, 18.71, 18.72, 19.18, 18.78, 19.03, 19.05, 18.77, 19.13])
# prog3d.package(mass=[18.76, 19.01, 19.09, 19.19])
# prog3d.clean(mass=[18.76, 19.01, 19.09, 19.19])

# prog3d.gen(mass=[19.11, 18.74, 19.16, 19.15, 19.02, 18.73], mode = 'detailed')
# prog3d.package(mass=[19.11, 19.16, 19.02, 18.73, 19.15, 18.74])
# prog3d.clean(mass=[19.11, 19.16, 19.02, 18.73, 19.15, 18.74])
#
# prog3d.gen(mass=[19.07, 19.04, 19.06, 19.08, 18.75, 19.12], mode = 'detailed')
# prog3d.package(mass=[19.07, 19.04, 19.06, 19.08, 18.75, 19.12])
# prog3d.clean(mass=[19.07, 19.04, 19.06, 19.08, 18.75, 19.12])
#
# prog3d.gen(mass=((np.arange(1,10)/100) + np.linspace(18,18.6,7)[:,np.newaxis]).reshape(-1))
# prog3d.package(mass=((np.arange(1,10)/100) + np.linspace(18,18.6,7)[:,np.newaxis]).reshape(-1))
# prog3d.clean(mass=((np.arange(1,10)/100) + np.linspace(18,18.6,7)[:,np.newaxis]).reshape(-1))
#
# prog3d.gen(mass=((np.arange(1,10)/100) + np.linspace(19.2,20.9,18)[:,np.newaxis]).reshape(-1))
# prog3d.package(mass=((np.arange(1,10)/100) + np.linspace(19.2,20.9,18)[:,np.newaxis]).reshape(-1))
# prog3d.clean(mass=((np.arange(1,10)/100) + np.linspace(19.2,20.9,18)[:,np.newaxis]).reshape(-1))
#
# prog3d.gen(mass=np.linspace(17.5,17.99,50))
# prog3d.package(mass=np.linspace(17.5,17.99,50))
# prog3d.clean(mass=np.linspace(17.5,17.99,50))
#
# prog3d.gen(mass=np.linspace(17.0,17.49,50))
# prog3d.package(mass=np.linspace(17.0,17.49,50))
# prog3d.clean(mass=np.linspace(17.0,17.49,50))
#
# prog3d.gen(mass=np.linspace(21.01,22,100))
# prog3d.package(mass=np.linspace(21.01,22,100))
# prog3d.clean(mass=np.linspace(21.01,22,100))
#
# prog3d.gen(mass=np.linspace(22.01,23,100))
# prog3d.package(mass=np.linspace(22.01,23,100))
# prog3d.clean(mass=np.linspace(22.01,23,100))
#
# prog3d.gen(mass=np.linspace(23.01,24,100))
# prog3d.package(mass=np.linspace(23.01,24,100))
# prog3d.clean(mass=np.linspace(23.01,24,100))
#
# prog3d.gen(mass=np.linspace(24.01,25,100), run=False)
# prog3d.package(mass=np.linspace(24.01,25,100))
# prog3d.clean(mass=np.linspace(24.01,25,100))
#
# prog3d.gen(mass=np.linspace(16.99,16,100), run=False)
# prog3d.package(mass=np.linspace(16.99,16,100))
# prog3d.clean(mass=np.linspace(16.99,16,100))
#
# prog3d.gen(mass=np.linspace(15.99,15,100), run=False)
# prog3d.package(mass=np.linspace(15.99,15,100))
# prog3d.clean(mass=np.linspace(15.99,15,100))
#
# prog3d.gen(mass=np.linspace(14.99,12,300), run=False)
#
# prog3d.gen(mass=np.linspace(25.01,27,200), run=False)
# prog3d.package(mass=np.linspace(25.01,25.56,56), run=False)
# prog3d.package(mass=np.linspace(25.57,26,44), run=False)
#
# prog3d.gen(mass=np.linspace(27.01,30,300), run=False)
#
# prog3d.gen(mass=np.linspace(30.01,32,200), run=False)
# prog3d.gen(mass=np.linspace(32.01,33,100), run=False)
# prog3d.gen(mass=np.linspace(33.01,35,200), run=False)
# prog3d.gen(mass=np.linspace(35.01,36,100), run=False)
# prog3d.gen(mass=np.linspace(36.01,40,400), run=False)

# =======================================================================
# low-mass series.
# prog3d.gen(mass=np.linspace(11.99,11,100), run=False, yeburn=True)
# prog3d.gen(mass=np.linspace(10.99,10,100), run=False, yeburn=True)
# prog3d.gen(mass=np.linspace(9.99,9,100), run=False, yeburn=True)
# prog3d.gen(mass=np.linspace(8.99,8,100), run=False, yeburn=True, after_file='/home/alex/kepler/prog3d/s9/s9g')
# prog3d.gen(mass=np.linspace(41,50,10), run=False)
# prog3d.gen(mass=np.linspace(51,60,10), run=False)
# prog3d.gen(mass=np.linspace(65,120,12), run=False)
