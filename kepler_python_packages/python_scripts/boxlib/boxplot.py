"""
Provide a version of PlotFile that can easily make plots using matplotlib

Laurens Keek 2011
"""

from . import boxlib
from pylab import *
import numpy as n

class PlotFile(boxlib.PlotFile):
    """
    A version of boxlib.PlotFile that can easily make quick plots
    """
    
    def get_xy_mesh(self, level=None):
        """
        Return spatial coordinates of zone boundaries
        """
        if level==None:
            level = self.level
        grid_spacing = self.grid_spacing[level]
        prob_lo = self.prob_lo
        prob_hi = self.prob_hi
        xy = []
        
        for dim in range(self.dimensions):
            xy.append(np.arange(prob_lo[dim],
                                prob_hi[dim] + grid_spacing[dim],
                                grid_spacing[dim]))
        return xy
    
    def plot(self, name, level=None, log=False): 
        """
        Color plot of field with given name at given level
        """
        if level==None:
            level = self.level
        
        x, y = self.get_xy_mesh(level)
        z = self.get(name, level).transpose()
        if log:
            z = n.log10(z)
        
        figure()
        im = pcolormesh(x, y, z)
        cb = colorbar(im, shrink=0.8)
        if log:
            cb.set_label('log(%s)'%name)
        else:
            cb.set_label(name)
        xlabel('X (cm)')
        ylabel('Y (cm)')
        suptitle('%s at level %i'%(self.filename, level))
    
    def plot_contour(self, name, level=None, log=False): 
        """
        Contour plot of field with given name at given level
        """
        if level==None:
            level = self.level
        
        x, y = self.get_xy(level)
        z = self.get(name, level).transpose()
        if log:
            z = n.log10(z)
        
        figure()
        cs = contour(x, y, z)
        clabel(cs, inline=True, fmt='%g')
        xlabel('X (cm)')
        ylabel('Y (cm)')
        if log:
            title('log(%s)'%name)
        else:
            title(name)
        suptitle('%s at level %i'%(self.filename, level))

    def plot_meanx(self, name, level=None, log=False):
        """
        Line plot of average value of field with <name> as a function of y
        """
        if level==None:
            level = self.level
        
        x, y = self.get_xy(level)
        z = self.get(name, level).mean(1)
        
        figure()
        if log:
            semilogy(x, z)
        else:
            plot(x,z)
        xlabel('X (cm)')
        ylabel('<%s>'%name)
        suptitle('%s at level %i'%(self.filename, level))
    
    def plot_meany(self, name, level=None, log=False):
        """
        Line plot of average value of field with <name> as a function of y
        """
        if level==None:
            level = self.level
        
        x, y = self.get_xy(level)
        z = self.get(name, level).mean(0)
        
        figure()
        if log:
            semilogy(y, z)
        else:
            plot(y,z)
        xlabel('Y (cm)')
        ylabel('<%s>'%name)
        suptitle('%s at level %i'%(self.filename, level))
    
    
