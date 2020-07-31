"""
Package to build f2py modules
"""

import os
import os.path
import subprocess
import importlib
import sys
import shutil
import itertools

from importlib.machinery import EXTENSION_SUFFIXES

class _Build():
    """
    Class to build module binaries.

    Having this in a class and not storing instance keeps namespace clean.
    """

    f2py_options = (
         f'f2py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
         f'f2py{sys.version_info.major}.{sys.version_info.minor}',
         f'f2py{sys.version_info.major}',
         f'f2py'
         )
    for f in f2py_options:
        if shutil.which(f):
            f2py_exec = f
            break
    else:
        raise Exception('f2py not found.')

    f2pycomp = f'{f2py_exec} --verbose'
    fcomp = 'gfortran -v'

    # These two need to be overwritten
    path = os.path.dirname(__file__)
    package = __package__

    sources = ('{}.f'.format(package),)
    objects = ('{}.o'.format(package),)
    libraries = ()
    include_paths = ()
    project_libraries = ()
    module = '_{}'.format(package)
    signature_file = '{}.pyf'.format(module)
    compile_flags = ('-fPIC',
                     '-O3',
                     '-funroll-loops',
                     '-fno-second-underscore',
                     '-fconvert=big-endian',
                     )

    executable = False
    executable_file = '{}.exe'.format(package)
    executable_link_flags = ()

    # does not yet work
    # from numpy.f2py import f2py2e
    # def f2py(self, args):
    #     result = self.f2py2e.run_main(args)
    #     print(result)

    def f2py(self, args):
        args = ([*(self.f2pycomp.split())]
             + list(args))
        result = subprocess.run(
            args,
            check = True,
            shell = False)
        print(result)

    def f2bin(self, args):
        result = subprocess.run(
            ([*(self.fcomp.split())]
             + list(args)),
            shell = True,
            check = True)
        print(result)

    def __init__(self):
        """
        init routine, likely to be called before doing own initialisations
        """
        # self.path = os.path.dirname(__file__)
        # self.package = os.path.basename(self.path)


    def run(self):
        """
        execute tests and build
        """
        if self.build_library_check() or self.build_check():
            self.build_module()

    def test_executable(self):
        """
        CUSTOM - test exectuable, raise error if problem
        """
        try:
            result = subprocess.run(os.path.join(self.path, self.executable_file), check = True)
        except subprocess.CalledProcessError:
            raise Exception("module executable failed")
        print(result)
        # TODO - a real test

    def make_executable(self):
        """
        build an executable a test case
        """
        try:
            for s,o in zip(self.sources, self.objects):
                self.f2bin([
                    '-c', s, '-o', o,
                    *self.compile_flags,
                    *('-I' + p for p in self.include_paths),
                    ])
            self.f2bin([
                *self.objects,
                *self.project_libraries,
                *self.libraries,
                *self.executable_link_flags,
                '-o', self.executable_file,
                ])
        except subprocess.CalledProcessError:
            raise Exception("executable compilation failed")

    def clean_executable(self):
        """
        remove executable file
        """
        try:
            os.remove(self.executable_file)
        except FileNotFoundError:
            pass


    def build_module(self):
        """
        Build python module binary library.

        We also do a test of the executable version
        """

        cwd = os.getcwd()
        os.chdir(self.path)

        if self.executable:
            # build executable
            self.make_executable()

            # test executable
            self.test_executable()

            # remove executable
            self.clean_executable()

        try:
            self.f2py([
                '-m', self.module,
                '-h', self.signature_file,
                '--include-paths', ':'.join(self.include_paths),
                *self.sources,
                '--overwrite-signature',
               ])
        except subprocess.CalledProcessError:
            raise Exception("creating f2py signature failed")
        try:
            self.f2py([
                '--f90flags={}'.format(
                    ' '.join(
                        itertools.chain(
                            self.compile_flags,
                            ('-I' + p for p in self.include_paths),
                            ),
                        ),
                    ),
                '--f77flags={}'.format(
                    ' '.join(
                        itertools.chain(
                            self.compile_flags,
                            ('-I' + p for p in self.include_paths),
                            ),
                        ),
                    ),
                '--include-paths', ':'.join(self.include_paths),
                *self.libraries,
                '-c',
                '-m', self.module,
                *self.project_libraries,
                *self.sources,
                self.signature_file,
                ])
        except subprocess.CalledProcessError:
            raise Exception("creating module failed")
        os.chdir(cwd)

    def build_library_check(self, debug = True):
        """
        CUSTOM check whether required libraries are up to date

        return value is whether library needs to be built
        """
        return False

    def build_check(self, debug = True):
        """
        check whether build is OK
        """
        so_file_base = os.path.join(self.path, self.module)

        for extension in EXTENSION_SUFFIXES:
            so_file = so_file_base + extension
            if os.path.exists(so_file):
                break
        else:
            if debug:
                print(' [DEBUG] so file does not exist.')
            return True

        source_files = [os.path.join(self.path, s) for s in self.sources]
        source_files += list(self.project_libraries)
        so_file_date = os.path.getctime(so_file)
        for f in source_files:
            if (so_file_date < os.path.getctime(f)):
                if debug:
                    print(' [DEBUG] {} newer than {}.'.format(f, so_file))
                return True
        try:
            importlib.import_module('.' + self.module, self.package)
        except ImportError as e:
            if debug:
                print(' [DEBUG] Import Error:', e)
                # sys.exit()
            return True

        # check for changed compiler version
        # (works on Fedora 18+)
        # other updates are welcome!
        # seems to have chned with gcc 9.0
        try:
            result = subprocess.check_output("gcc --version", shell=True).decode('ASCII', errors='ignore')
            compiler_version = (result.splitlines()[0]).split(' ', 2)[2]
            result = subprocess.check_output("strings - " + so_file + " | grep GCC:", shell = True).decode('ASCII', errors='ignore')
            library_version = []
            library_version.append(result.splitlines()[0].split(' ', 2)[2]) # pre 9.0
            library_version.append(result.splitlines()[0].split('GCC:')[1].split(' ', 2)[2]) # 9.1
            if not compiler_version in library_version:
                if debug:
                    print(' [DEBUG] compiler/library version mismatch:')
                    print(" [DEBUG] Compiler Version {}".format(compiler_version))
                    print(" [DEBUG]  Library Version {}".format(library_version))
                return True
        except:
            print(" [DEBUG] Compiler comparison failed.")
            return True
        return False
