"""
Convience wrapper around _helmholtz for using F.X.Timmes's Helmholtz EOS in python

Implemented by Laurens Keek

See helmholtz.helmholtz.py for documentation

COMPILE INSTRUCTIONS

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

name = 'helmholtz'

def _build_setup():
    import os.path

    global path, so_file_base

    path = os.path.dirname(__file__)
    
    so_file_base = os.path.join(path, '_' + name )

# import sysconfig
# so_file = os.path.join(path,'_' + name + 
#                        sysconfig.get_config_var('SO'))
def _build_module():
    """
    Build the Helmholtz FOTRTAN module.

    We also do a test of the executable version
    """

    import os
    import subprocess
    import glob
    import numpy as np

    cwd = os.getcwd()
    os.chdir(path)

    f2py_signature = 'f2py3' # f2py because the f2py3 has a bug here

  # /usr/lib64/python3.3/site-packages/numpy/f2py/crackfortran.py
  # line 2609
  #           exec('c = isintent_%s(var)' % intent)
  # by
  #           c = eval('isintent_{:s}(var)'.format(intent))

# patch -R /usr/lib64/python3.3/site-packages/numpy/f2py/crackfortran.py -
# --- /usr/lib64/python3.3/site-packages/numpy/f2py/crackfortran.py       2013-04-25 17:31:54.355737972 +1000
# +++ /usr/lib64/python3.3/site-packages/numpy/f2py/crackfortran.py       2013-04-11 02:48:20.000000000 +1000
# @@ -2606,7 +2606,7 @@
#      ret = []
#      for intent in lst:
#          try:
# -            c = eval('isintent_{:s}(var)'.format(intent))
# +            exec('c = isintent_%s(var)' % intent)
#          except NameError:
#              c = 0
#          if c:

    f2py_library   = 'f2py3'

    # test executable
    try:
        subprocess.check_call('gfortran '+name+'.f90 -o ' + name + '.exe',shell=True)
    except subprocess.CalledProcessError:
        print("executable compilation failed")
        raise
    try:
        result = subprocess.check_output('./' + name + ".exe").decode()
    except subprocess.CalledProcessError:
        print("module executable failed")
        raise
    print(result)
    a = np.array([float(x) for x in result.split()[26:28]])
    print(a)
    b = np.array([6.96100959E+22,  1.02965110E+17])
    print('Test result OK?: ', np.all(np.equal(a,b)))
    try:
        os.remove(name + ".exe")
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

        subprocess.check_call(f2py_signature + ' -m _'+name+' -h _'+name+'.pyf '+name+'.f90 --overwrite-signature',shell=True)
    except subprocess.CalledProcessError:
        print("creating f2py signature failed")
        raise
    try:
        subprocess.check_call(f2py_library + " --f90exec=/usr/bin/gfortran --f90flags='-fPIC -O3 -funroll-loops -fno-second-underscore' -L. -lgfortran -c -m _"+name+" "+name+".f90",shell=True)
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

    f90_file = os.path.join(path, name+'.f90')
    if (os.path.getctime(so_file) <
        os.path.getctime(f90_file)):
        if debug:
            print('[DEBUG] so file too old.')
        return True
    try:
        # ipython auto-reload seems to have an issue with this
        # when the packages is reloaded and *this* file has changed.
        from . import _helmholtz
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

__all__ = ['helmholtz']


from .helmholtz import Helmholtz
