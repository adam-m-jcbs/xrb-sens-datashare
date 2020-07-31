import numpy as np
import matplotlib.pyplot as plt
import os
class Plots(object):
    def __init__(self, masses, time, polynomial, correctedtime, fracinterp,  massfrac, c, egy, IMF):
        """
        This module contains all of the plotting functions
        """
        self.masses = masses
        self.time = time
        self.logmasses = np.log(masses)
        self.logtime = np.log(time)
        self.tnew = np.linspace(np.log(self.time[0]),np.log(self.time[-1]),1000)
        self.polynomial = polynomial
        self.deriv = polynomial.deriv()
        self.correctedtime = correctedtime
        self.fracinterp = fracinterp
        self.massfrac = massfrac  
        self.c = c
        self.egy = egy
        self.xinterp = np.linspace(self.correctedtime[0], self.correctedtime[-1], 10000)
        self.IMF = IMF
        
    # For plotting mass vs time (raw data)              
    def masstimeplot(self):
        """
        Plots the raw mass v time data
        """
        fig = plt.figure(1)
        ax = fig.add_subplot(1,2,1)
        ax.plot(self.masses, (self.time))
        ax.set_xlabel(r'$M/\mathrm{M}_\odot$')
        ax.set_ylabel(r'Time to Supernova (s)')
        ax.set_title(r'Time to supernova for early stars of varying mass')
        
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(np.log(self.masses), np.log(self.time))
        ax2.set_xlabel(r'$M/\mathrm{M}_\odot$')
        ax2.set_ylabel(r'Time to Supernova (s)')
        ax2.set_title(r'Log Log plot')
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.expanduser('~/python/project/outputfiles/masstimeplot.pdf'), bbox_inches = 'tight')	

###########################################################################    

    # Plotting the interpolation functions       
    def interpolationfunctionplot(self):
        """
        Plots the interpolated mass v time function
        """
        fig = plt.figure(1)
        
        ax = fig.add_subplot(1,2,1)
        ax.plot(self.time,self.masses, 'o',  np.exp(self.tnew), np.exp(self.polynomial(self.tnew)), '-') 
        ax.set_ylabel(r'$M/\mathrm{M}_\odot$')
        ax.set_xlabel(r'Time to Supernova (s)')
        ax.set_title(r'Interpolated function')
        
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(self.logtime,self.logmasses, 'o', self.tnew, self.polynomial(self.tnew), '-')
        ax2.set_ylabel(r'ln $M/\mathrm{M}_\odot$')
        ax2.set_xlabel(r'ln Time to Supernova (s)')
        ax2.set_title(r'Interpolated function (log)')
        
        y = np.exp(self.tnew) #regular data
        x = np.exp(self.polynomial(self.tnew))
        
        dx = x[1:] - x[:-1] 
        dy = y[1:] - y[:-1]
        ax = 0.5*(x[1:] + x[:-1]) #average
        ay = 0.5*(y[1:] + y[:-1])
        n = dy*ax/(dx*ay)  #y=x**n, n is local gradient dt/dm
        
        fig = plt.figure(2)
        
        a1 = fig.add_subplot(1,1,1)
        a1.plot(ax,n, '-')  
        a1.set_title('Evolution of n with increasing mass')
        a1.set_xlabel('$M/\mathrm{M}_\odot$')
        a1.set_ylabel('n')
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.expanduser('~/python/project/outputfiles/interpfunc.pdf'), bbox_inches = 'tight')
############################################################################        
   
    # this plots the interpolated function with the correct mass and times       
    def correctedtimeplot(self):
        """
        Plots mass v the adjusted time fixed by the root finding fixtime function
        """
        fig = plt.figure(1)
        
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.correctedtime,self.masses, 'o',  np.exp(self.tnew), np.exp(self.polynomial(self.tnew)), '-') # plots original data with interpolation function data once its been raised to exponential to return it to original form
        ax.set_ylabel('$M/\mathrm{M}_\odot$')
        ax.set_xlabel('Time to Supernova (s)')
        ax.set_title('Interpolated function and corrected times')
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.expanduser('~/python/project/outputfiles/correctedtime.pdf'), bbox_inches = 'tight')
#############################################################################        
    # plot of the mass fractions over time    
    def massfracplot(self):
        """
        Plots the mass fractions (ejecta mass over initial mass) of each star
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.correctedtime, self.massfrac, 'o', self.xinterp, self.fracinterp(self.xinterp), '-')
        ax.set_ylabel('Mass ejection Fraction')
        ax.set_xlabel('Time to Supernova (s)')
        ax.set_title('Mass ejection fraction for corrected times')
        fig.tight_layout()
        plt.show()                
        fig.savefig(os.path.expanduser('~/python/project/outputfiles/massfrac.pdf'), bbox_inches='tight')      


    
######################################################
     
 
        
    # this plots the big function we are integrating to obtain yield solutions    
    def functiontointegrateplot(self):
        """
        Plots the main mass function which is integrated to obtain ejecta
        """
        m = lambda t: np.exp(self.polynomial(np.log(t)))
        g = lambda t: self.IMF(m(t))*self.c*self.deriv(np.log(t))*self.fracinterp(t)/t*m(t)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.correctedtime, g(self.correctedtime), '-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mass ejection')
        ax.set_title('Plot of mass ejection to be integrated')
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.expanduser('~/python/project/outputfiles/integrationfunction.pdf'), bbox_inches = 'tight')

#######################################################
    # plots the explosion energy as a function of mass    
    def egymassplot(self):
        """
        Plots the supernova explosion energy as calculated by Heger and Mueller
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.masses, self.egy, '.')
        ax.set_xlabel('Initial Mass (Solar Masses)')
        ax.set_ylabel('Explosion Energy (B)')
        ax.set_title('Supernova Explosion energy')
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.expanduser('~/python/project/outputfiles/explosionenergy.pdf'), bbox_inches = 'tight')
