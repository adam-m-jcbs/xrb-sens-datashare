"""
Module to handle git repository creation

used for inpurt from tar.gz and cvs
"""

import re, os, os.path
import subprocess, shutil, glob, calendar
import contextlib
import time
import tarfile
import tempfile
import gzip
import shutil

from logged import Logged
from human import time2human
from human import byte2human
from utils import iterable

from utils import environ, walk_files

dir0 = os.path.expanduser('~/git/')
base = os.path.join(dir0, 'prog')

os.environ['TZ'] = 'Australia/Melbourne'

def check_dates():
    files = sorted(glob.glob('/home/alex/git/prog/KEPLER????????.tar.gz'))
    times = []
    for f in files:
        s = re.findall('KEPLER(\d+)\.',f)[0]
        filedate = time.strptime(s,'%Y%m%d')
        file_sec = calendar.timegm(filedate)
        t = Tar(f)
        tdate = t.get_last_date()
        s2 = time.strftime('%Y%m%d', time.gmtime(tdate))
        OK = '--'
        ok = '  '
        x = ''
        times += [tdate]
        if len(times) > 1:
            if times[-2] < file_sec:
                OK = 'OK'
            if times[-2] <  tdate:
                ok = 'OK'
            elif times[-2] >  tdate:
                ok = '##'
                x = '{:>6.2f}'.format((times[-2] - tdate) / 86400),
        print(s, s2,
              '{:>6.2f}'.format((file_sec - tdate) / 86400),
              OK, ok, x)

class IDL(Logged):
    def __init__(self):
        self.target_dir = os.path.expanduser('~/git/IDL')
        self.base = os.path.expanduser('~/git/cvs/IDL')
        modules = next(os.walk(self.base))[1]
        print(modules)
        # ['astrolib', 'Collapse', 'Stern', 'license', 'mhd', 'common', 'ICF', 'hydro', 'stern', 'distribution', 'ps', 'MPEG', 'flash', 'AMR', 'data', 'color', 'bin', 'jet', 'Nucleosynthesis', 'VV', 'PS', 'EZ', 'doc', 'starfit', 'thesis', 'NumSim']
        return
        try:
            shutil.rmtree(self.target_dir)
        except:
            pass

        try:
            shutil.rmtree('/home/alex/.cvsps')
        except:
            pass

        os.mkdir(self.target_dir)
        for m in modules:
            self.convert(m)


    def convert(self, module):
        # make repository
        os.chdir('/home/alex/git')
        args = 'git cvsimport -a -v -A /home/alex/git/author-conv-file.txt -pxv -z 3600 -R -C'.split(' ')
        args += ['IDL/' + module]
        args += ['-d']
        args += ['/home/alex/git/cvs']
        args += ['IDL/' + module ]
        subprocess.check_call(args)

        # update branches and tags
        os.chdir('/home/alex/git/IDL/' + module)
        subprocess.check_call(['git', 'branch', '-D', 'origin'])
        subprocess.check_call(['git', 'tag', 'cvsimport'])

        if os.path.isfile('.cvsignore'):
            print('updating ignored files')
            os.rename('.cvsignore', '.gitignore')
            subprocess.check_call(['git', 'add', '.gitignore'])
            subprocess.check_call(['git', 'commit', '-am', 'rename .cvsignore to .gitignore'])

        subprocess.check_call(['git', 'gc', '--prune=now', '--aggressive'])

        os.chdir('/home/alex/git')
        subprocess.check_call(['gitg', 'IDL/' + module])
        subprocess.check_call(['git', 'clone', '--bare', 'IDL/' + module, 'IDL/' + module + '.git'])
        subprocess.check_call('rsync -e ssh -vazu --progress --delete'.split(' ') + ['IDL/' + module + '.git', 'a:temp/IDL'])


class Run(Logged):
    def __init__(self):
        files = []
        for f in walk_files(base):
            if f.endswith('tar.gz'):
                files += [f]
        self.files = files

        for x in files:
            print(x)

def date2str(date):
    date = time.localtime(date)
    s = '{:04d}{:02d}{:02d}'.format(
        date.tm_year,
        date.tm_mon,
        date.tm_mday)
    return s

def copy_content(source, target, exclude_dirs = None):
    dirpath, dirs, files = next(os.walk(source))
    for f in files:
        s = os.path.join(dirpath, f)
        t = os.path.join(target, f)
        print(' ... copying {} to {}'.format(s, t))
        shutil.copy2(s, t, follow_symlinks = False)
    for d in dirs:
        if not d in iterable(exclude_dirs):
            s = os.path.join(dirpath, d)
            t = os.path.join(target, d)
            print(' ... copying directory {} to {}'.format(s, t))
            shutil.copytree(s, t, symlinks = True)

class Git(Logged):
    """
    Class to interact with git repository

    Needs path to repsitory directory.
    """

    git_committer_environ = ('GIT_COMMITTER_DATE', 'GIT_COMMITTER_EMAIL', 'GIT_COMMITTER_NAME')
    git_prog_path = shutil.which('git')

    def __init__(self, path,
                 init = False,
                 new = False,
                 pull = False,
                 checkout = None,
                 clone = None,
                 detach = True,
                 ):
        path = os.path.expanduser(os.path.expandvars(path))
        self.path = path
        if clone is not None:
            new = True
            pull = False
        if new:
            try:
                shutil.rmtree(path)
            except:
                pass
            init = True
        if clone:
            if isinstance(clone, Git):
                clone = clone.path
            # cannot use env because dir does not exist
            subprocess.check_call(['git', 'clone', clone, path])
            if detach is True:
                self.git_command(['remote', 'remove', 'origin'])
        else:
            if not os.path.isdir(path):
                os.mkdir(path)
            if init:
                self.init()
        if checkout is not None:
            self.checkout(checkout)
        if pull is True:
            self.pull()

    @contextlib.contextmanager
    def chdir(self):
        cwd = os.getcwd()
        os.chdir(self.path)
        yield
        os.chdir(cwd)

    @classmethod
    def clear_committer_environ(cls):
        for env in (cls.git_committer_environ):
            try:
                del os.environ[env]
            except:
                pass

    def git_command(self, args):
        with self.chdir():
            subprocess.check_call([self.git_prog_path] + args)

    def git_output(self, args):
        with self.chdir():
            return subprocess.check_output([self.git_prog_path] + args, universal_newlines = True)

    def reset(self, ref, hard = True):
        args = ['reset']
        if hard:
            args += ['--hard']
        args += [ref]
        self.git_command(args)

    def init(self):
        self.git_command(['init'])

    def branch(self, branch, ref = None, checkout = False):
        args = ['branch', branch]
        if ref is not None:
            args += [ref]
        self.git_command(args)
        if checkout == True:
            self.checkout(branch)

    def checkout(self, branch):
        self.git_command(['checkout', branch])

    def get_hash(self,
                 ref = 'HEAD',
                 before = None,
                 after = None):
        args = ['log', '-1', '--pretty=%H']
        if before is not None:
            args += ['--before', "{}".format(before)]
        if after is not None:
            del args['-1']
            args += ['--after', "{}".format(after)]
        args += [ref]
        refs = self.git_output(args)
        if after:
            return refs.splitlines()[0]
        else:
            return refs.strip()

    def get_revlist(self, ref = 'HEAD', base = None):
        if base is not None:
            ref = ref + '...' + base
        args = ['rev-list', ref]
        return self.git_output(args).splitlines()

    def get_initial_rev(self, checkout = False, **args):
        rev = self.get_revlist(**args)[-1]
        if checkout:
            self.chechout(rev)
        return rev

    def find_next(self, ref, branch = 'HEAD'):
        refs = self.get_revlist(ref = branch, base = ref)
        if len(refs) > 0:
            return refs[-1]

    def pull(self, remote = None):
        args = ['pull']
        if remote is not None:
            args += [remote]
        self.git_command(args)

    def commit(self,
               date_string = None,
               message = None,
               email = 'alex@ucolick.org',
               name = 'Alexander Heger',
               info = None
               ):

        if info is None:
            if message is None:
                message = date_string
            info = dict()
            info['message'] = message
            info['committer_name'] = name
            info['committer_email'] = email
            info['committer_date'] = date_string
            info['author_name'] = name
            info['author_email'] = email
            info['author_date'] = date_string

        env = {
            'GIT_COMMITTER_DATE' : info['committer_date' ],
            'GIT_COMMITTER_EMAIL': info['committer_email'],
            'GIT_COMMITTER_NAME' : info['committer_name' ],
            }
        if not self.is_clean():
            with environ(env):
                self.git_command(['add', '--all'])
                self.git_command(
                    ['commit',
                     '--date="{author_date}"'.format(**info),
                     '--author="{author_name} <{author_email}>"'.format(**info),
                     '-m', info['message']
                     ])
        else:
            print(' *** Nothing to commit. ***')
        return self.get_hash()

    def get_status(self):
        return self.git_output(['status'])

    def undo_commit(self):
        branch = self.get_branch()
        self.git_command(['reset','^HEAD'])
        self.git_command(['update-ref', 'refs/heads/' + banch, 'HEAD'])
        self.git_command(['gc', '--prune=now'])

    def is_clean(self):
        return len(re.findall(r'\nnothing to commit, working directory clean\n',
                     self.get_status())) == 1

    def get_commit_info(self, ref = 'HEAD'):
        args = ['cat-file', 'commit', ref]
        lines = self.git_output(args).splitlines()
        tree = lines[0][5:]
        parents = []
        i = 1
        while lines[i].startswith('parent '):
            parents += [lines[i][7:]]
            i += 1
        author_info = lines[i]
        committer_info = lines[i+1]
        message = '\n'.join(lines[i+3:])

        extract = re.compile('[a-z]+ (.*) <(.*)> ([0-9]+ .*)')
        author_name, author_email, author_date = extract.findall(author_info)[0]
        committer_name, committer_email, committer_date = extract.findall(committer_info)[0]

        return dict(
            message = message,
            author_name = author_name,
            author_email = author_email,
            author_date = author_date,
            committer_name = committer_name,
            committer_email = committer_email,
            committer_date = committer_date,
            tree = tree,
            parents = parents,
            )

    def get_commit_time(self, ref = 'HEAD'):
        args = ['show', '-s', '--format=%ct', ref]
        return float(self.git_output(args))

    def update_committer_from_author(self, base = None, branch = None):
        args = ['filter-branch',
                '--commit-filter',
                'export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"; export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"; export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"; git commit-tree "$@"',
                '--',
                base + '..HEAD']
        self.checkout(branch)
        self.git_command(args)
        args = ['update-ref', '-d', 'refs/original/refs/heads/' + branch ]
        self.git_command(args)

    def rebase(self, branch, ref = None, onto = None, root = None):
        args = ['rebase', '--committer-date-is-author-date']
        if onto is not None:
            args += ['--onto', onto]
        if root is not None:
            args += ['--root', root]
        if ref is not None:
            args += [ref]
        args += [branch]
        self.git_command(args)

    def remove_all_tags(self):
        tags = self.git_output(['tag', '-l']).splitlines()
        self.git_command(['tag', '-d'] + tags)

    def remove_tag(self, name):
        args = ['tag', '-d', name]
        self.git_command(args)

    def tag(self, name, ref = None, overwrite = True):
        if overwrite:
            tags = self.git_output(['tag', '-l', name]).splitlines()
            if len(tags) == 1:
                print(' *** Overwriting tag {} ***'.format(name))
                self.remove_tag(name)
        args = ['tag', name]
        if ref is not None:
            args += [ref]
        self.git_command(args)

    def remove_branch(self, name):
        args = ['branch', '-D', name]
        self.git_command(args)

    def merge(self, name):
        args = ['merge', name]
        self.git_command(args)

    def remove_remote(self, name):
        args = ['remote', 'remove', name]
        self.git_command(args)

    def gc(self, prune = False):
        args = ['gc', '--aggressive']
        if prune is True:
            args += ['--prune=now']
        self.git_command(args)

    def clone(self, path, bare = False):
        if os.path.isdir(path):
            shutil.rmtree(path)
        args = ['clone']
        if bare is True:
            args += ['--bare']
        args += ['.', path]
        self.git_command(args)

    def add_remote(self, name, remote, fetch = False):
        assert isinstance(remote, Git)
        self.git_command(['remote', 'add', name, remote.path])
        if fetch is True:
            self.git_command(['fetch', name])

    def clean(self):
        dirpath, dirs, files = next(os.walk(self.path))
        for f in files:
            os.remove(os.path.join(dirpath, f))
        for d in dirs:
            p = os.path.join(dirpath, d)
            if os.path.islink(p):
                os.remove(p)
            else:
                if d != '.git':
                    print('deleting ',d)
                    shutil.rmtree(p)

    def copy_from(self, other, clean = False):
        if isinstance(other, Git):
            path = other.path
        else:
            path = os.path.expanduser(os.path.expandvars(other))
        if clean is True:
            self.clean()
        copy_content(path, self.path, exclude_dirs = ('.git',))
        # dirpath, dirs, files = next(os.walk(path))
        # for f in files:
        #     s = os.path.join(dirpath, f)
        #     t = os.path.join(self.path, f)
        #     print(' ... copying {} to {}'.format(s, t))
        #     shutil.copy2(s, t, follow_symlinks = False)
        # for d in dirs:
        #     if d != '.git':
        #         s = os.path.join(dirpath, d)
        #         t = os.path.join(self.path, d)
        #         print(' ... copying directory {} to {}'.format(s, t))
        #         shutil.copytree(s, t, symlinks = True)

    def get_branch(self):
        branches = self.git_output(['branch']).splitlines()
        for b in branches:
            if b.startswith('*'):
                branch = b[2:]
        if branch.startswith('{'):
            branch = re.findall(' ([a-f0-9])+}')[0]
        return branch

    def replace_from(self, other, hash = None, commit = True, info = None):
        if hash is not None:
            branch = other.get_branch()
            # add staching?
            other.checkout(hash)
        self.clean()
        self.copy_from(other)
        if commit:
            ix = other.get_commit_info()
            if info is not None:
                ix.update(info)
            self.commit(info = ix)
        if hash is not None:
            other.checkout(branch)

    def rebase_other(self, other,
                     branch = 'master',
                     other_branch = 'master',
                     tmp_branch = 'x',
                     tmp_remote = 'X',
                     initial = None,
                     ):

        print('#'*72)
        print('*** Rebasing from ', other.path)

        self.checkout(branch)
        self.add_remote(tmp_remote, other, fetch = True)
        self.branch(tmp_branch, ref = tmp_remote + '/' + other_branch)
        if initial is None:
            initial = other.get_initial_rev()
        self.replace_from(other, hash = initial, commit = True)

        refs = self.get_revlist(ref = tmp_branch, base = initial)
        if len(refs) == 0:
            print(' *** Nothing to rewrite *** ')
        else:
            self.rebase(tmp_branch, ref = initial, onto = branch)
            self.update_committer_from_author(base = branch, branch = tmp_branch)
            self.checkout(branch)
            self.merge(tmp_branch)
        self.remove_remote(tmp_remote)
        self.remove_branch(tmp_branch)




class Tar(Logged):
    def __init__(self, filename):
        self.tar = tarfile.open(filename,'r:*', bufsize=2**25)

        # last_date = self.get_last_date()
        # time_string = date2str(last_date)

        self.filename = self.tar.fileobj.filename
        try:
            self.short = re.findall('(\d{8})',self.filename)[-1]
        except:
            short = None

    def get_file_date(self, branch = False):
        filedate = time.strptime(self.short, '%Y%m%d')
        file_sec = calendar.timegm(filedate)
        return file_sec

    def get_date_string(self):
        # s = self.short
        # return '{:s}-{:s}-{:s}T12:00:00'.format(s[0:4],s[4:6],s[6:8])
        return time.asctime(time.gmtime(self.get_file_date() + 12*60*60))

    def get_last_date(self):
        t = []
        for m in self.tar.getmembers():
            if m.isfile():
                t += [m.mtime]
        return max(t)

    def extract(self, git, files = None, clean = True, subdir = None):
        assert isinstance(git, Git)
        if clean:
            git.clean()
        if files is None:
            extract = None
        else:
            regs = [re.compile(f) for f in iterable(files)]
            extract = []
            members = self.tar.getmembers()
            for r in regs:
                extract += [m for m in members if r.fullmatch(m.name) is not None]
            print(extract)
        if subdir is not None:
            with tempfile.TemporaryDirectory() as temp:
                self.tar.extractall(path = temp, members = extract)
                copy_content(os.path.join(temp, subdir), git.path)
        else:
            self.tar.extractall(path = git.path, members = extract)

    def clean_kepler(self, path):
        for f in walk_files(path, ignore = '.git'):
            delete = False
            b = os.path.basename(f)
            if b.endswith(('.so','.a','.o','.pdf','.ps', '.buggy', '~', '.dvi', '.toc', '.log', '.aux', '#')):
                delete = True
            if b in ('keplery', 'noalps','core','xvista.tar.gz', 'specl.f.1', 'Ratenew.f', 'mixing.save'):
                delete = True
            if b == 'grid.ps':
                delete = False
            if delete:
                os.remove(f)
                print(' --> Removing ', f)

        if os.path.isdir(os.path.join(path, 'doc')):
            os.rename(os.path.join(path, 'doc'), os.path.join(path, 'addon'))
            print(' --> Renaming doc --> addon')
        if os.path.isdir(os.path.join(path, 'man')):
            shutil.rmtree(os.path.join(path, 'man'))
            print(' --> Removing man')


    def submit(self,
               git,
               date = None,
               info = None,
               files = None,
               clean = None,
               subdir = None,
               presubmit = None,
               ):
        assert isinstance(git, Git)
        self.extract(git, files = files, subdir = subdir)
        if clean == 'kepler':
            self.clean_kepler(git.path)
        args = dict()
        if isinstance(date, str):
            args['date_string'] = date
        elif isinstance(info, dict):
            args['info'] = info
        else:
            try:
                args['date_string'] = self.get_date_string()
            except:
                raise Exception('Need Commit date.')
        if presubmit is not None:
            presubmit()
        return git.commit(**args)

def unzip(infile, outdir = None, outfile = None):
    assert infile.endswith('.gz')
    if outfile is None:
        outfile = infile[:-3]
        if outdir is not None:
            outfile = os.path.join(outdir, os.path.basename(outfile))

    with gzip.open(infile, 'rb') as f_in, open(outfile, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


class Stern():
    tardir = '/home/alex/Stern/Save/prog'
    datatar = '/home/alex/Stern/Save/data'
    stardir = '/home/alex/git/Stern/source'
    datadir = os.path.join(stardir, 'data')
    def __init__(self):

        try:
            os.mkdir('/home/alex/git/Stern')
            os.mkdir('/home/alex/git/Stern/source')
        except:
            pass


        files = sorted(glob.glob(os.path.join(self.tardir, 'Stern???????????.tar.gz')))
        files += sorted(glob.glob(os.path.join(self.tardir, 'Stern????????.tar.gz')))
        files += sorted(glob.glob(os.path.join(self.tardir, 'STERN????????.tar.gz')))
        test = sorted(glob.glob(os.path.join(self.tardir, 'Stern????????TEST.tar.gz')))[0]

        git = Git(self.stardir, new = True, init = True)
        self.git = git

        data = glob.glob(os.path.join(self.datatar, '*.gz'))

        def file_add():
            if not os.path.isdir(self.datadir):
                os.mkdir(self.datadir)
            for d in data:
                unzip(d, outdir = self.datadir)

        first = None
        self.refs = dict()
        for f in files:
            t = Tar(f)
            commit = t.submit(self.git, clean = 'kepler', presubmit = file_add)
            self.refs[t.short] = commit
            if first is None:
                first = commit

        # print(reflist)
        print('### making branches ###')

        self.add_branch('TEST-19990307',
                        '19990224',
                        (test,
                         ),
                        )

        git.checkout('master')

        git.tag('historic', first)
        git.tag('tar-end')

        git.gc(prune = True)
        git.clone('/home/alex/git/Stern.git', bare = True)
        shutil.rmtree('/home/alex/git/Stern')

    def add_branch(self, name, base, files):
        h = self.refs[base]
        self.git.branch(name, ref = h, checkout = True)
        for f in files:
            t = Tar(f)
            commit = t.submit(self.git, clean = 'kepler')

class SternW():
    tardir = '/home/alex/stern'
    stardir = '/home/alex/git/stern'
    def __init__(self):

        try:
            os.mkdir('/home/alex/git/stern')
        except:
            pass

        files = sorted(glob.glob(os.path.join(self.tardir, 'stern????????.tar.gz')))

        git = Git(self.stardir, new = True, init = True)
        self.git = git

        first = None
        self.refs = dict()
        for f in files:
            t = Tar(f)
            commit = t.submit(self.git)
            self.refs[t.short] = commit
            if first is None:
                first = commit

        git.checkout('master')

        git.tag('original', first)
        git.tag('tar-end')

        git.gc(prune = True)
#        git.clone('/home/alex/git/stern.git', bare = True)
#        shutil.rmtree('/home/alex/git/stern')

class Kepler():
    tardir = '/home/alex/git/prog'
    kepdir = '/home/alex/git/kepler/source'
    def __init__(self):

        try:
            os.mkdir('/home/alex/git/kepler')
            os.mkdir('/home/alex/git/kepler/source')
        except:
            pass

        files = sorted(glob.glob(os.path.join(self.tardir, 'KEPLER????????.tar.gz')))
        git = Git(self.stardir, new = True, init = True)
        self.git = git

        first = None
        self.refs = dict()
        for f in files:
            t = Tar(f)
            commit = t.submit(self.git, clean = 'kepler')
            self.refs[t.short] = commit
            if first is None:
                first = commit

        # print(reflist)
        print('### making branches ###')

        self.add_branch('BURN-500',
                        '19980420',
                        ('KEPLER19980421-500',
                         ),
                        )


        self.add_branch('woosley-weak',
                        '20000218',
                        ('KEPLER20000220W',
                         'KEPLER20000220W-EC20000523',
                         'KEPLER20000220W-EC20000531',
                         'KEPLER20000220W-EC20000707',
                         ),
                        )

        self.add_branch('fxt-neutrino',
                        '20000308',
                        ('KEPLER20000308-20000521',
                         'KEPLER20000308-20000717',
                         ),
                        )

        self.add_branch('C12ag',
                        '20000828',
                        ('KEPLER20000828-20001125',
                         ),
                        )

        self.git.checkout('master')
        hcvs = git.get_hash()

        # link in (rebase) old tranlated repository
        Git.clear_committer_environ()

        # hbase = '57cf59b610e3416c62c29351df593058482f534f'
        # other = Git('/home/alex/git/kc')
        hbase = '93dbe8010f6e3fb5d51f80891a310eeff395a973'
        other = Git('/home/alex/git/kf')
        try:
            other.checkout('master')
            other.remove_branch('cvs')
        except:
            pass
        other.branch('cvs', ref = hbase, checkout = True)

        git.replace_from(other)
        git.commit(date_string = 'Sat May 25 04:57:11 UTC 2002')

        git.add_remote('CVS', other, fetch = True)
        git.branch('cvs', ref = 'CVS/master', checkout = False)
        git.remove_all_tags()
        git.rebase('cvs', ref = 'CVS/cvs', onto = 'master')
        git.checkout('cvs')
        git.update_committer_from_author(base = 'master', branch = 'cvs')
        git.checkout('master')
        cvs_hash = git.get_hash()
        git.merge('cvs')
        git.remove_remote('CVS')
        git.remove_branch('cvs')
        git.tag('historic', first)
        git.tag('cvs-start', cvs_hash)
        git.tag('cvs-end')

        hlast = git.get_hash()
        hold = 'cvs-end'

        new = Git('/home/alex/kepler/source', checkout = 'master')
        new_commits = new.get_revlist(base = hold, ref = 'master')
        for commit in new_commits[::-1]:
            new.checkout(commit)
            info = new.get_commit_info()
            info['author_email'] = 'alexander.heger@monash.edu'
            info['committer_email'] = 'alexander.heger@monash.edu'
            git.replace_from(new)
            git.commit(info = info)
        new.checkout('master')

        git.gc(prune = True)
        git.clone('/home/alex/git/kepler.git', bare = True)
        shutil.rmtree('/home/alex/git/kepler')

    def add_branch(self, name, base, files):
        h = self.refs[base]
        self.git.branch(name, ref = h, checkout = True)
        for f in files:
            t = Tar(os.path.join(self.tardir, f + '.tar.gz'))
            commit = t.submit(self.git, clean == 'kepler')

class Python():
    def __init__(self, clean = False):
        """ to be done ... """
        os.chdir('/home/alex/git')
        if clean:
            try:
                shutil.rmtree('/home/alex/git/python/source2')
            except:
                pass
            try:
                shutil.rmtree('/home/alex/git/python/source3')
            except:
                pass
            try:
                shutil.rmtree('/home/alex/git/python/sourcex')
            except:
                pass
            try:
                shutil.rmtree('/home/alex/.cvsps')
            except:
                pass
            try:
                os.mkdir('/home/alex/git/python')
            except:
                pass
            os.system('git cvsimport -a -v -A ~/git/authors-python2.txt -pxv -R -C ~/git/python/source2 -d /home/alex/git/cvs python/source')
            os.system('git cvsimport -a -v -A ~/git/authors-python3.txt -pxv -R -C ~/git/python/source3 -d /home/alex/git/cvs python/source3')
            pyx = Git('/home/alex/git/python/sourcex', clone = 'git@a:python/source', detach= False)
            pyx.branch('python-2.7', ref = 'origin/python-2.7')
        else:
            pyx = Git('/home/alex/git/python/sourcex', pull = True)
        py2 = Git('/home/alex/git/python/source2')
        py3 = Git('/home/alex/git/python/source3')
        py = Git('/home/alex/git/python/source', clone = py2)

        branch = py.get_hash(before = '2013-04-01')
        py.branch('python-2.7')
        py.tag('end-cvs-python-2.7')
        initial = py.tag('initial', py.get_initial_rev())
        py.tag('end-develop-python-2.7', branch)
        py.reset(branch)

        # add py3
        py.rebase_other(py3, tmp_branch = 'p3', tmp_remote = 'P3')
        py.tag('end-cvs')
        py.tag('after-2to3', py.find_next(branch)
)

        # get updates from pyx for master
        py.rebase_other(pyx, initial = pyx.get_hash('end-cvs'))
        py.tag('git', py.find_next('end-cvs'))

        # get updates from pyx for python2.7
        py.rebase_other(pyx,
                        branch='python-2.7',
                        other_branch='python-2.7',
                        initial = pyx.get_hash('end-cvs-python-2.7'))
        py.tag('git-python-2.7', py.find_next('end-cvs-python-2.7',
                                              branch = 'python-2.7'))
        py.checkout('master')
        py.gc(prune = True)
        py.clone('/home/alex/git/python.git', bare = True)
        os.chdir('/home/alex/git')
        shutil.rmtree('/home/alex/git/python')

class Batch(Logged):
    tardir = '/home/alex/git/BATCH'
    batdir = '/home/alex/git/batch'
    def __init__(self, clean = False):
        os.chdir('/home/alex/git')

        if clean:
            try:
                shutil.rmtree('/home/alex/git/kepler/batch1')
            except:
                pass
            try:
                shutil.rmtree('/home/alex/git/kepler/batch2')
            except:
                pass
            try:
                shutil.rmtree('/home/alex/git/kepler/batch3')
            except:
                pass
            try:
                shutil.rmtree('/home/alex/.cvsps')
            except:
                pass
            try:
                os.mkdir('/home/alex/git/kepler')
            except:
                pass
            os.system('git cvsimport -a -v -A ~/git/author-conv-file.txt -pxv -z 3600 -R -C kepler/batch1 -d /home/alex/git/cvs kepler/batch')
            os.system('git cvsimport -a -v -A ~/git/author-conv-file.txt -pxv -z 3600 -R -C kepler/batch2 -d /home/alex/git/cvs kepler/batch2')
            os.system('git cvsimport -a -v -A ~/git/author-conv-file.txt -pxv -z 3600 -R -C kepler/batch3 -d /home/alex/git/cvs kepler/batch3')

        batch1 = Git('/home/alex/git/kepler/batch1')
        batch2 = Git('/home/alex/git/kepler/batch2')
        batch3 = Git('/home/alex/git/kepler/batch3')

        # make repository
        batch = Git(self.batdir, new = True, init = True)

        # add tar files
        files = sorted(glob.glob(os.path.join(self.tardir, 'BATCH????????.tar.gz')))
        refs0 = []
        for f in files:
            t = Tar(f)
            commit = t.submit(batch)
            refs0 += [commit]

        batch.tag('initial', refs0[0])

        batch.rebase_other(batch1, tmp_branch = 'b1', tmp_remote = 'B1')
        batch.branch('batch')

        initial = batch.find_next(refs0[-1])
        batch.tag('cvs', initial)

        batch.rebase_other(batch2, tmp_branch = 'b2', tmp_remote = 'B2')
        batch.branch('batch2')

        initial = batch.find_next('batch')
        batch.tag('batch2-initial', initial)

        batch.rebase_other(batch3, tmp_branch = 'b3', tmp_remote = 'B3')
        batch.branch('batch3')

        initial = batch.find_next('batch2')
        batch.tag('batch3-initial', initial)

        batch.tag('cvs-end')
        batch.gc(prune = True)
        batch.clone('/home/alex/git/batch.git', bare = True)
        shutil.rmtree('/home/alex/git/kepler')


def format_time(t):
    return time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(t))

def get_dir_file_date_string(path):
    t = 0.
    path = os.path.expanduser(os.path.expandvars(path))
    for f in os.listdir(path):
        f = os.path.join(path, f)
        if os.path.isfile(f):
            t = max(t, os.path.getmtime(f))
    return format_time(t)

class Develop(Logged):
    def __init__(self):
        os.chdir('/home/alex/git')
        try:
            shutil.rmtree('/home/alex/.cvsps')
        except:
            pass
        try:
            os.mkdir('/home/alex/git/kepler')
        except:
            pass
        dirs = (
            'neutrino-fxt',
            'newrate',
            'n14ec',
            'nu_process',
            'nuebarcap',
            'weaknu',
            'light_rates',
            )
        for d in dirs:
            try:
                shutil.rmtree(os.path.join('/home/alex/git/kepler', d))
            except:
                pass
            os.system('git cvsimport -a -v -A ~/git/author-conv-file.txt -px,v -z 3600 -R -C ~/git/kepler/{0} -d /home/alex/git/cvs kepler/develop/{0}'.format(d))
            git = Git(os.path.join('/home/alex/git/kepler', d))
            git.remove_branch('origin')
            initial = git.get_initial_rev()
            git.tag('cvs', initial)
            git.tag('cvs-end')
            git.gc(prune = True)
            git.clone(os.path.join('/home/alex/git', d + '.git'), bare = True)
        shutil.rmtree('/home/alex/git/kepler')

class KepVer(Logged):
    def __init__(self, new = False):
        os.chdir('/home/alex/git')
        if new:
            try:
                shutil.rmtree('/home/alex/git/kepler')
            except:
                pass
            clone = 'git@a:kepler/source.git'
        else:
            clone = None
        git = Git('/home/alex/git/kepler/source', clone = clone)
        revs = git.get_revlist('master')
        re_version1 = re.compile(r'subroutine\s+gener.*setparm\s*=\s*([-+0-9\.DdEe]+)\s*\n',
                                flags = re.DOTALL)
        for r in revs[::-1]:
            git.checkout(r)
            with open(os.path.join(git.path, 'kepou.f'),'rt') as f:
                text = f.read()
            version = re_version1.findall(text)
            info = git.get_commit_info()
            if len(version) == 0:
                break
            print(time.asctime(time.localtime(int(info['author_date'].split(' ')[0]))), version)
        print(r)
        # r = '17f0cca37a81544f4fc82df59f7082c40c7fbf38'
        i = revs.index(r)
        re_version2 = re.compile(
            r'^\s+parameter\s+\(\s*currentversion\s*=\s*([-+0-9\.DdEe]+)\s*\)\s*$',
                                flags = re.MULTILINE)
        for r in revs[i::-1]:
            git.checkout(r)
            with open(os.path.join(git.path, 'kepcom'),'rt') as f:
                text = f.read()
            version = re_version2.findall(text)
            info = git.get_commit_info()
            if len(version) == 0:
                break
            print(time.asctime(time.localtime(int(info['author_date'].split(' ')[0]))), version)
        # r = 'e1e8f9158434fe135d317661a6eaa12b7e9db097'
        i = revs.index(r)
        re_version3 = re.compile(
            r'^\s+parameter\s+\(\s*ncurversion\s*=\s*([-+0-9\.DdEe]+)\s*\)\s*$',
                                flags = re.MULTILINE)
        for r in revs[i::-1]:
            git.checkout(r)
            with open(os.path.join(git.path, 'kepcom'),
                      mode = 'rt',
                      encoding = 'ascii',
                      errors = 'ignore') as f:
                text = f.read()
            version = re_version3.findall(text)
            info = git.get_commit_info()
            if len(version) == 0:
                break
            print(time.asctime(time.localtime(int(info['author_date'].split(' ')[0]))), version)

class Mongo(Logged):
    def __init__(self, clean = True):
        """ to be done ... """
        os.chdir('/home/alex/git')
        if clean:
            try:
                shutil.rmtree('/home/alex/.cvsps')
            except:
                pass
            try:
                shutil.rmtree('/home/alex/git/kepler')
            except:
                pass
            try:
                os.mkdir('/home/alex/git/kepler')
            except:
                pass
            os.system('git cvsimport -a -v -A ~/git/author-conv-file.txt -pxv -R -C ~/git/kepler/source -d /home/alex/git/cvs_xvista kepler/source')
            os.system('git cvsimport -a -v -A ~/git/author-conv-file.txt -pxv -R -C ~/git/kepler/mongo_dp -d /home/alex/git/cvs kepler/mongo_dp')
            os.system('git cvsimport -a -v -A ~/git/author-conv-file.txt -pxv -R -C ~/git/kepler/mongo_dp64 -d /home/alex/git/cvs kepler/mongo_dp64')
            # os.system('git cvsimport -a -v -A ~/git/authors-python3.txt -pxv -R -C ~/git/kepler/mongo_gcc -d /home/alex/git/cvs_mongo kepler/mongo_gcc')
            # os.system('git cvsimport -a -v -A ~/git/authors-python3.txt -pxv -R -C ~/git/kepler/mongo_intel -d /home/alex/git/cvs kepler/mongo_intel')
            # os.system('git cvsimport -a -v -A ~/git/authors-python3.txt -pxv -R -C ~/git/kepler/mongo_intel_mac -d /home/alex/git/cvs kepler/mongo_intel_mac')

        mongo = Git('/home/alex/git/kepler/mongo', new = True)

        # add original files
        org = Tar('/home/alex/git/prog/KEPLER19961212.tar.gz')
        org.extract(mongo, files = 'libmongo\..*')
        org = Tar('/home/alex/git/MONGO/Mongo_Src/fonts.tar.gz')
        org.extract(mongo, clean = False)
        org = Tar('/home/alex/git/MONGO/Mongo_Src/mongo.tar.gz')
        org.extract(mongo, clean = False)
        mongo.commit(date_string = get_dir_file_date_string(mongo.path))
        mongo.branch('medusa')
        mongo.tag('initial')

        # add some MPA files
        mongo.branch('MPA', checkout = True)
        mongo.clean()
        MPA_path = '~/git/MONGO/MPA'
        mongo.copy_from(MPA_path)
        mongo.commit(date_string = get_dir_file_date_string(MPA_path))
        mongo.checkout('master')

        tars = Git('~/git/kepler/source')
        revs = tars.get_revlist('master')
        refs = []
        for r in revs[:0:-1]:
            t = tars.checkout(r)
            t = Tar('/home/alex/git/kepler/source/xvista.tar.gz')
            commit = t.submit(mongo, info = tars.get_commit_info(r))
            refs += [commit]
        mongo.branch('xvista')

        dp = Git('~/git/kepler/mongo_dp')
        dp64 = Git('~/git/kepler/mongo_dp64')
        gcc = Git('~/git/kepler/mongo_gcc')
        intel = Git('~/git/kepler/mongo_intel')
        intel_mac = Git('~/git/kepler/mongo_intel_mac')

        mongo.rebase_other(dp, tmp_branch = 'xdp', tmp_remote = 'xDP')
        mongo.branch('dp')
        mongo.tag('dp-end-cvs')


        # dp --> dp64 (branch at initial date
        #
        dp_initial_time = dp64.get_commit_time(dp64.get_initial_rev())
        rev = mongo.get_hash(before = format_time(dp_initial_time))
        mongo.branch('dp64', rev, checkout = True)
        mongo.rebase_other(dp64, branch = 'dp64', tmp_branch = 'xdp64', tmp_remote = 'xDP64')
        mongo.tag('dp64-end-cvs')

        mongo.branch('gcc_mac')
        mongo.branch('intel_mac_libs', checkout = True)
        mongo.copy_from('~/git/MONGO/mongo_intel_mac', clean = True)
        mongo.commit(date_string = "Sat May  5 03:40:06 UTC 2012",
                     email='alexander.heger@monash.edu')
        mongo.tag('intel_mac_libs-end-cvs')

        mongo.checkout('gcc_mac')
        mongo.copy_from('~/git/MONGO/mongo_dong', clean = True)
        mongo.commit(date_string = "Sun Jul 21 12:00:00 UTC 2013",
                     email='alexander.heger@monash.edu')

        mongo.copy_from('~/git/MONGO/mongo_gcc_mac', clean = True)
        mongo.commit(date_string = "Tue Jul 23 12:00:00 UTC 2013",
                     email='alexander.heger@monash.edu')
        mongo.tag('gcc_mac-end-cvs')

        mongo.branch('intel', checkout = True)
        mongo.copy_from('~/git/MONGO/mongo_intel', clean = True)
        mongo.commit(date_string = "Thu Jul 25 12:00:00 UTC 2013",
                     email='alexander.heger@monash.edu')
        mongo.tag('intel-end-cvs')

        mongo.checkout('gcc_mac')
        mongo.branch('gcc', checkout = True)
        mongo.copy_from('~/git/MONGO/mongo_gcc', clean = True)
        mongo.commit(date_string = "Fri Jul 26 13:40:56 UTC 2013",
                     email='alexander.heger@monash.edu')
        mongo.tag('gcc-end-cvs')

        mongo.remove_branch('master')
        mongo.branch('master', checkout = True)
        mongo.tag('cvs-end')

        mongo.checkout('gcc_mac')
        t = Tar('/home/alex/git/MONGO/mongo_gcc_mac_20140508.tar.gz')
        t.submit(mongo,
                 date = "Thu May 07 19:15:00 PDT 2014",
                 subdir = 'mongo_gcc')

        mongo.checkout('intel_mac_libs')
        t = Tar('/home/alex/git/MONGO/mongo_intel_mac_20140508.tar.gz')
        t.submit(mongo,
                 date = "Thu May 07 19:15:01 PDT 2014",
                 subdir = 'mongo_intel')

        mongo.checkout('master')

        mongo.gc(prune = True)
        mongo.clone('/home/alex/git/mongo.git', bare = True)
        shutil.rmtree('/home/alex/git/kepler')


if __name__ == '__main__':
    pass
