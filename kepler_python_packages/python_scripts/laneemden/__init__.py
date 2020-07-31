"""
Lane Emden Routines

f2py -m _solver -h _solver.pyf solver.f90 --overwrite-signature
f2py --f90exec=/usr/bin/gfortran --f90flags='-fPIC -O3 -funroll-loops -fno-second-underscore' -L. -lgfortran -c -m _solver solver.f90

gfortran solver.f90 -o solver
solver

This may require to create a link to libgefortran.so

ln -s /usr/lib64/libgfortran.so.3 libgfortran.so

Or, for all users:
[root]# cd /usr/lib64
[root]# ln -s libgfortran.so.3 libgfortran.so

"""

def _build_setup():
    import os.path

    global path, so_file_base

    path = os.path.dirname(__file__)

    so_file_base = os.path.join(path, '_solver')

# import sysconfig
# so_file = os.path.join(path,'_solver' +
#                        sysconfig.get_config_var('SO'))
def _build_module():
    """
    Build the Lane Emden FORTRAN module.

    We also do a test of the executable version
    """

    import os
    import subprocess
    import glob
    import numpy as np
    import sys
    import shutil

    cwd = os.getcwd()
    os.chdir(path)

    f2py_options = (
         'f2py{}.{}.{}'.format(sys.version_info.major,sys.version_info.minor,sys.version_info.micro),
         'f2py{}.{}'.format(sys.version_info.major,sys.version_info.minor),
         'f2py{}'.format(sys.version_info.major),
         'f2py'
         )
    for f in f2py_options:
        if shutil.which(f):
            f2py_exec = f
            break
    else:
        raise Exception('f2py not found.')

    f2py_library   = f2py_exec
    f2py_signature = f2py_exec

    # test executable
    try:
        subprocess.check_call("gfortran solver.f90 -o solver.exe",shell=True)
    except subprocess.CalledProcessError:
        print("executable compilation failed")
        raise
    try:
        result = subprocess.check_output("./solver.exe")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("module executable failed")
        raise
    a = np.array([float(x) for x in result.split()])
    b = np.array([6.8968486193938672,0.,-4.24297576045549740e-2])
    print('Test result OK?: ',np.all(np.equal(a,b)))
    try:
        os.remove("solver.exe")
    except:
        pass

    # make module
    libs = glob.glob('/usr/lib64/libgfortran.so*')
    if len(libs) > 0:
        try:
            os.stat("libgfortran.so")
        except:
            try:
                os.remove("libgfortran.so")
            except:
                pass
            lib = libs[0]
            print('Trying to link ' + lib)
            try:
                os.symlink(lib, "libgfortran.so")
            except:
                print("Could not create link")
    try:

        subprocess.check_call(f2py_signature + " -m _solver -h _solver.pyf solver.f90 --overwrite-signature",shell=True)
    except subprocess.CalledProcessError:
        print("creating f2py signature failed")
        raise
    try:
        subprocess.check_call(f2py_library + " --f90exec=/usr/bin/gfortran --f90flags='-fPIC -O3 -funroll-loops -fno-second-underscore' -L. -lgfortran -c -m _solver solver.f90",shell=True)
    except subprocess.CalledProcessError:
        print("creating module failed")
        raise
    os.chdir(cwd)

def _build_check(debug = True):
    """
    check whether build is OK
    """
    import os.path
    import subprocess

    from importlib.machinery import EXTENSION_SUFFIXES

    global so_file_base, so_file

    for extension in EXTENSION_SUFFIXES:
        so_file = so_file_base + extension
        if os.path.exists(so_file):
            break
    else:
        if debug:
            print('[DEBUG] so file does not exist.')
        return True

    f90_file = os.path.join(path, 'solver.f90')
    if (os.path.getctime(so_file) <
        os.path.getctime(f90_file)):
        if debug:
            print('[DEBUG] so file too old.')
        return True
    try:
        # ipython auto-reload seems to have an issue with this
        # when the packages is reloaded and *this* file has changed.
        from . import _solver
    except ImportError as e:
        if debug:
            print('[DEBUG] Import Error:', e)
        return True

    # check for changed compiler version (works on Fedora 18+)
    try:
        result = subprocess.check_output("gcc --version", shell=True).decode('ASCII', errors='ignore')
        compiler_version = (result.splitlines()[0]).split(' ', 2)[2]
        result = subprocess.check_output("strings - " + so_file + " | grep GCC", shell = True).decode('ASCII', errors='ignore')
        library_version = result.splitlines()[0].split(' ', 2)[2]
        if compiler_version != library_version:
            if debug:
                print('[DEBUG] compiler/library version mismatch:')
                print("[DEBUG] Compiler Version {}".format(compiler_version))
                print("[DEBUG]  Library Version {}".format(library_version))
            return True
    except:
        print("[DEBUG] Compiler comparison failed.")
    return False


def _build_clean():
    global path, so_file
    del path, so_file
    global _build_setup, _build_check, _build_module, _build_clean
    del _build_setup, _build_check, _build_module, _build_clean

_build_setup()
if _build_check():
    _build_module()
_build_clean()

__all__ = ['solver']

from .solver import lane_emden_int, lane_emden_step
