"""
Convience wrapper around _helmholtz for using F.X.Timmes's Helmholtz EOS in python

Generating _helmholtz from the fortran source using f2py:
f2py chokes on line 214 of helmholtz.f90. Change to:
     parameter        (sioncon = (2.0d0 * pi * amu * kerg/h/h), &
Create the _helmholtz.so file using gfortran:
f2py --f90exec=/usr/bin/gfortran --f90flags='-fPIC -O3 -funroll-loops -fno-second-underscore' -lgfortran -c -m _helmholtz helmholtz.f90

This may require to create a link to libgefortran.so

[root]# cd /usr/lib64
[root]# ln -s libgfortran.so.3 libgfortran.so

Implemented by Laurens Keek

Example:
import helmholtz
h = helmholtz.Helmholtz()
h.temp = 1e8;h.den=1e6;h.abar=2;h.zbar=1
h.eos()
h.out()
"""

import os
import helmholtz._helmholtz as _helmholtz
import numpy as n

# Load data table
old = os.getcwd() 
os.chdir(os.path.dirname(__file__))
_helmholtz.read_helm_table()
os.chdir(old)

class Helmholtz():
    """
    Convenience class for helmholtz eos calls
    """

    def __init__(self):
        """
        Initialize
        """
        self.jlo_eos = 1
        self.jhi_eos = 1
        self.temp = 1e8
        self.den = 1e6
        self.abar = 2
        self.zbar = 1
    
    def eos(self):
        """
        Perform the eos call
        """
        _helmholtz.eosvec2.jlo_eos = self.jlo_eos
        _helmholtz.eosvec2.jhi_eos = self.jhi_eos
        # Copy input variables
        _helmholtz.thinp.abar_row[0] = self.abar 
        _helmholtz.thinp.zbar_row[0] = self.zbar
        _helmholtz.thinp.temp_row[0] = self.temp
        _helmholtz.thinp.den_row[0] = self.den
        _helmholtz.thinp.abar_row[0] = self.abar
        _helmholtz.thinp.abar_row[0] = self.abar
        
        _helmholtz.helmeos()

        # Copy calculated quantities
        self.pres = _helmholtz.ptotc1.ptot_row[0] 
        self.ener = _helmholtz.etotc1.etot_row[0]
        self.entr = _helmholtz.stotc1.stot_row[0]
        self.pgas = _helmholtz.thpgasc1.pgas_row[0]
        self.egas = _helmholtz.thegasc2.egas_row[0]
        self.sgas = _helmholtz.thsgasc1.sgas_row[0]
        self.prad = _helmholtz.thprad.prad_row[0]
        self.erad = _helmholtz.therad.erad_row[0]
        self.srad = _helmholtz.thsrad.srad_row[0]
        self.pion = _helmholtz.thpion.pion_row[0]
        self.eion = _helmholtz.theion.eion_row[0]
        self.sion = _helmholtz.thsion.sion_row[0]
        self.xni = _helmholtz.th_xni_ion.xni_row[0]
        self.cs = _helmholtz.thdergc2.cs_gas_row[0]
        self.cp = _helmholtz.thdergc1.cp_gas_row[0]
        self.cv = _helmholtz.thdergc1.cv_gas_row[0]
        
        # derivatves
        self.dpt = _helmholtz.ptotc1.dpt_row[0]
        self.dpd = _helmholtz.ptotc1.dpd_row[0]
    
    def out(self):
        """
        Call _helmholtz.pretty_eos_out. Only valid directly after calling eos()
        """
        _helmholtz.pretty_eos_out(self.jlo_eos)
