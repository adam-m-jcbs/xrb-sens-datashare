
import os
import os.path
import subprocess
import glob
import uuid
import hashlib
import re
import glob
import shutil

import cpuinfo

from collections import OrderedDict

from ..build import _Build

def _versionfunc(versionflags, machine = None, hashing = False):
    flags = versionflags.copy()
    if machine == 'MAC':
        flags['MAC'] = f'{uuid.getnode():012x}'.upper()
    elif machine == 'CPU':
        # we could add 'stepping', 'model', 'family'
        # can't use entire CPU info as it contains current clock frequency
        cpu = cpuinfo.get_cpu_info()['brand']
        cpu = re.sub('[^[a-zA-Z0-9]+', '_', cpu)
        flags['CPU'] = cpu
    elif machine is not None:
        raise Exception ('machine type not supported')
    if hashing:
        version = hashlib.sha1(str(flags).encode()).hexdigest().upper()
    else:
        version = '_'.join(f'{k}_{v}' for k, v in flags.items())
    return version


def clean():
    path = os.path.dirname(__file__)
    print(f' [CLEAN] Clenaing files in  {path}')
    files = glob.glob(os.path.join(path, '_kepler*'))
    for f in files:
        if os.path.isfile(f):
            print(f' [CLEAN] Deleting file      {f}')
            os.unlink(f)
        else:
            print(f' [CLAEN] Removing directory {f}')
            shutil.rmtree(f)
    # shutil.rmtree(os.path.join(path), '__pycache__')

class _BuildKepler(_Build):

    package = __package__
    path = os.path.dirname(__file__)

    sources = ('kepler.f90', 'data.f')
    objects = ('kepler.o', 'data.o')
    libraries = ('-luuid',)

    # executable_link_flags = ('-fconvert=big-endian',)
    executable_link_flags = ()
    compile_flags = _Build.compile_flags + executable_link_flags

    def __init__(
            self,
            NBURN = None,
            JMZ = None,
            FULDAT = None,
            NAME = None,
            machine = 'CPU',
            hashing = False,
            update = True,
            ):
        """
        Note - all initilisation needed is done in class definition for now.
        Maybe that should go here instead ...
        """

        buildflags = OrderedDict()

        if NBURN is not None:
            buildflags['NBURN'] = str(NBURN)
        if JMZ is not None:
            buildflags['JMZ'] = str(JMZ)
        if FULDAT is not None:
            buildflags['FULDAT'] = FULDAT
        if NAME is not None:
            buildflags['NAME'] = NAME

        self.buildflags = buildflags
        self.makeflags = [f'{k}={v}' for k, v in buildflags.items()]

        self.update = update

        versionflags = buildflags.copy()
        if 'FULDAT' in versionflags:
            versionflags['FULDAT'] = versionflags['FULDAT'].replace('.', '_')
        self.version = '_' + _versionfunc(versionflags, machine = machine, hashing = hashing)

        if len(versionflags) == 0:
            print('[build] building with defaults from Makefile.')

        # my own stuff
        self.kepler_source_path  = os.path.join(
            os.environ['KEPLER_PATH'],
            'source')
        self.kepler_library_path = os.path.join(
            self.path,
            f'_kepler{self.version}')
        self.kepler_library_file = os.path.join(
            self.kepler_library_path,
            'kepler.a')

        self.project_libraries = (
            self.kepler_library_file,
            )
        self.include_paths = (
            self.kepler_source_path,
            self.kepler_library_path,
            )
        self.signature_file = f'_kepler{self.version}.pyf'
        self.module = f'_kepler{self.version}'
        self.executable_file = 'kepler{self.version}.exe'

        super().__init__()

    def build_library_check(self, debug = True):
        """
        check whether KEPLER library is up to date
        """
        try:
            library_time = os.path.getctime(self.kepler_library_file)
            if self.update == False:
                return False
        except FileNotFoundError:
            library_time = 0

        exclude = ('uuidcom', 'gitcom', 'nburncom', 'gridcom', )
        patterns = ('*com', '*.f', '*.f90', '*.c', 'Makefile*', )
        makefile = os.path.join(self.path, 'Makefile')
        last_time = os.path.getctime(makefile)
        for p in patterns:
            for f in glob.glob(os.path.join(self.kepler_source_path, p)):
                if os.path.basename(f) in exclude:
                    continue
                last_time = max(last_time, os.path.getctime(f))
        if last_time > library_time:
            cwd = os.getcwd()
            try:
                os.mkdir(self.kepler_library_path)
            except FileExistsError:
                pass
            os.chdir(self.kepler_library_path)
            cmd = ['make', '-j', '-f', makefile] + self.makeflags
            subprocess.run(cmd, shell = False, check = True)
            os.chdir(cwd)
            return True
        return False
