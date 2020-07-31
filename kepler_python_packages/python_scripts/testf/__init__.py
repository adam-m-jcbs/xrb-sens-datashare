"""
Routine to test import of FORTRAN librray files, based on Lane Emden example

This may require to create a link to libgefortran.so

ln -s /usr/lib64/libgfortran.so.3 libgfortran.so

Or, for all users:
[root]# cd /usr/lib64
[root]# ln -s libgfortran.so.3 libgfortran.so

For Python3 there is a bug in numpy < 1.8 

FIX:

  /usr/lib64/python3.3/site-packages/numpy/f2py/crackfortran.py
  line 2609
            exec('c = isintent_%s(var)' % intent)
  by      
            c = eval('isintent_{:s}(var)'.format(intent))

OR:

patch -R /usr/lib64/python3.3/site-packages/numpy/f2py/crackfortran.py -
--- /usr/lib64/python3.3/site-packages/numpy/f2py/crackfortran.py       2013-04-25 17:31:54.355737972 +1000
+++ /usr/lib64/python3.3/site-packages/numpy/f2py/crackfortran.py       2013-04-11 02:48:20.000000000 +1000
@@ -2606,7 +2606,7 @@
     ret = []
     for intent in lst:
         try:
-            c = eval('isintent_{:s}(var)'.format(intent))
+            exec('c = isintent_%s(var)' % intent)
         except NameError:
             c = 0
         if c:

"""

def _build_setup():
    """
    set up build variables
    """
    import os.path
    import importlib.machinery

    global path, so_file, main, external, ext

    path = os.path.dirname(__file__)

    so_file = os.path.join(
        path,
        '_solver' + importlib.machinery.EXTENSION_SUFFIXES[0])

    external = ['rk4']
    main = 'solver'
    ext = '.f90'

# import sysconfig
# so_file = os.path.join(path,'_solver' +
#                        sysconfig.get_config_var('SO'))
def _build_module():
    """
    Build the Lane Emden FOTRTAN module.

    We also do a test of the executable version
    """
    
    import os
    import subprocess
    import glob
    import numpy as np

    cwd = os.getcwd()
    os.chdir(path)

    f2py_signature = 'f2py3'
    f2py_library   = 'f2py3'
    f90flags='-fPIC -O3 -funroll-loops -fno-second-underscore'

    # test executable
    try:
        subprocess.check_call("gfortran -c *"+ext+" "+f90flags,shell=True)
        subprocess.check_call("gfortran *.o -o "+main+".exe "+f90flags,shell=True)
    except subprocess.CalledProcessError:
        print("executable compilation failed")
        raise
    try:
        result = subprocess.check_output(main+".exe")
    except subprocess.CalledProcessError:
        print("module executable failed")
        raise
    a = np.array([float(x) for x in result.split()])
    b = np.array([6.8968486193938672,0.,-4.24297576045549740e-2])
    print('Test result OK?: ',np.all(np.equal(a,b)))
    try:
        os.remove(main+".exe")
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
            print('Trying to link '+lib)
            try:
                os.symlink(lib, "libgfortran.so")
            except:
                print("Could not create link")
    try:        
        subprocess.check_call(f2py_signature + " -m _"+main+" -h _"+main+".pyf "+main+ext+" --overwrite-signature",shell=True)
    except subprocess.CalledProcessError:
        print("creating f2py signature failed")
        raise
    try:        
        subprocess.check_call("gfortran -c " + ' '.join([e + ext for e in external]) + " " +
                              f90flags,
                              shell=True)
        subprocess.check_call(f2py_library +
                              " --f90exec=/usr/bin/gfortran --f90flags='" +
                              f90flags +
                              "' -L. -lgfortran -c -m _"+main+" "+main+ext+" " +
                              ' '.join([e + '.o' for e in external]),
                              shell=True)
    except subprocess.CalledProcessError:
        print("creating module failed")
        raise
    os.chdir(cwd)

def _build_check():
    """
    Check whether build is OK.
    """
    import os.path
    import subprocess

    if not os.path.exists(so_file):
        return True
    files = [os.path.join(path, main + ext)] + [os.path.join(path, e + ext) for e in external]    
    tso = os.path.getctime(so_file)
    for f in files:
        if (tso < os.path.getctime(f)):        
            return True
    try:
        from . import _solver
    except ImportError:
        return True

    # check for changed compiler version (works on Fedora 18)
    try:
        result = subprocess.check_output("gcc --version", shell=True)
        compiler_version = (result.splitlines()[0]).split(' ',2)[2]
        result = subprocess.check_output("strings - " + so_file + " | grep GCC", shell=True)
        library_version = result.splitlines()[0].split(' ',2)[2]
        if compiler_version != library_version:
            return True
    except:
        pass
    return False


def _build_clean():
    """
    Clean up build variables.
    """
    global path, so_file, external, main, ext
    del path, so_file, external, main, ext
    global _build_setup, _build_check, _build_module, _build_clean
    del _build_setup, _build_check, _build_module, _build_clean

_build_setup()
if _build_check():
    _build_module()
_build_clean()

__all__ = ['solver']

from .solver import lane_emden_int, lane_emden_step
