#!/usr/bin/env python
"""
A convenience wrapper around _load2d for loading 2d plot files from boxlib.

Build requirements: gfortran, f2py, numpy, python, a CASTRO installation.
To build the _load2d module, place the directory containing this file into
fParallel/data_processing of your CASTRO installation, and run make. This
may require to create a link to libgefortran.so:
ln -s -f /usr/lib64/libgfortran.so.3 libgfortran.so

Based on the fIDLdump.f90 by M. Zingale. Implemented by Laurens Keek 2011.
"""

from _load2d import load2d
import numpy as np

class PlotFile:
    """
    Convenience class for reading boxlib plot files.
    """
    
    def __init__(self, filename, level=1):
        """
        Initialize a new PlotFile using given filename.
        
        filename: path to a plotfile directory containing a Header file
        level: the AMR level at which data will be returned
        """
        self.filename = filename
        self.read_header() # Read header with python
        #self._read_header() # Read header with boxlib 
        self.set_level(level)
    
    def _read_header(self):
        """
        Read information from the Header file using boxlib. Currently
        not used! See read_header() for a python implementation.
        """
        load2d.load(filename, 1, '')
        self.time = float(load2d.time)
        self.names = self._get_names()
        self.max_level = int(load2d.max_available_level)
    
    def _get_names(self):
        """
        Get the available variable names from the current plot file
        """
        a = load2d.names
        b = a.transpose().reshape(a.shape)
        return [''.join(b[i]).strip() for i in range(b.shape[0])]
    
    def set_level(self, level):
        """
        Set the level at which data will be returned. If level is larger than
        the maximum available level, the latter is used.
        """
        self.level = min(level, self.max_level)
    
    def get(self, name, level=None):
        """
        Get the data for component with name. Optionally,
        specify the AMR level.
        """
        if level==None:
            level = self.level
        
        load2d.load(self.filename, level, name)
        if len(name)>0: # Empty name for reading header
            data = load2d.c_fab[:,:,0].copy()
            load2d.l2d_clear() # Free memory
            return data
    
    def get_xy(self, level=None):
        """
        Return spatial coordinates of zone centers.
        """
        if level==None:
            level = self.level
        grid_spacing = self.grid_spacing[level]
        prob_lo = self.prob_lo
        prob_hi = self.prob_hi
        xy = []

        for dim in range(self.dimensions):
            xy.append(np.arange(prob_lo[dim] + grid_spacing[dim]/2.0,
                                prob_hi[dim], grid_spacing[dim]))
        return xy
    
    def __repr__(self):
        """
        Return a string representation
        """
        return "boxlib.PlotFile('%s', level=%i)"%(self.filename, self.level)
    
    def read_header(self):
        """
        Read part of the header file. Alternative to having boxlib do this.
        """
        lines = open('%s/Header'%self.filename).readlines()
        nvars = int(lines[1])
        i = 2
        self.names = [line.strip() for line in lines[i:i+nvars]]
        i += nvars
        self.dimensions = int(lines[i])
        i += 1
        self.time = float(lines[i])
        i += 1
        self.max_level = int(lines[i]) + 1
        i += 1
        self.prob_lo = [float(part) for part in lines[i].split()]
        i += 1
        self.prob_hi = [float(part) for part in lines[i].split()]
        i += 1
        self.amr_ref_ratio = [int(part) for part in lines[i].split()]
        i += 1
        # skipping: Prob domains per level
        i += 2
        level = 1
        grid_spacing = {}
        for i in range(i, i + self.max_level):
            grid_spacing[level] = [float(part) for part in lines[i].split()]
            level += 1
        self.grid_spacing = grid_spacing
        # skipping: boxes per level; basically an index for the plot data accross files

if __name__ == '__main__':
    # List contents of given plot file
    import sys
    if len(sys.argv)>1:
        pf = PlotFile(sys.argv[1])
        print('Time:', pf.time)
        print('Maximum AMR level:', pf.max_level)
        print('Component names:', ', '.join(pf.names))
    else:
        print('Usage: boxlib.py <plotfile>')
