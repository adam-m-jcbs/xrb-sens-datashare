import numpy as np
import sn_analytic
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import stardb
import kepdump
import isotope
from scipy.interpolate import interp1d
from scipy.integrate import romberg, quad
import polyinterp, fixtime, rombergintervalsolver, yieldout, projectplots, makedata, remnantobtain, elementobtain, massyieldplotter, solutionsolve, isoyieldplt,  salpeter, re
#energy functions
def sn_energy_default(d):
    """
    This is calling Heger and Mueller's sn_analytic code to get the explosion energy.

    To use a different energy, uncomment out the energy function below and set as desired. Make sure to use reload=True also when running Main
    """
    sn=sn_analytic.SN()
    return sn.get_explosion(d)['e_expl']*1e-51
#def sn_energy_default(d):
#    return 1.2
#mixing function
def mixfunc(energy):
    """
    User can set any desired mixing value here. Useable values can be seen by loading the database or checking the mixingvalues parameter once the code has been run
    """
    return 0.1

class Main(object):
    def __init__(self,
                 dbfilename = os.path.expanduser('~/python/project/znuc2012.S4.star.el.y.stardb.gz'),
                 
                 efunc = sn_energy_default,
                 reload = False):
        """
        init will check if the module has been run before and if so, will
        load all required files and data.  If not, it will have to
        load the database and do all required interpolation and
        function solving which will take ~30 seconds. Applying
        reload=True will force the program to reload, do if energyfunc
        or mixing func was changed by the user

        """
        save_path = os.path.expanduser('~/python/project/filestoload/')
        output_path = os.path.expanduser('~/python/project/outputfiles/')                
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)          
        if (not reload) and os.path.isfile(os.path.expanduser('~/python/project/filestoload/Energyvalues.npy')):
            energyvalues = np.load(os.path.expanduser('~/python/project/filestoload/Energyvalues.npy'))
        
        if (not reload) and os.path.isfile(os.path.expanduser('~/python/project/filestoload/Mixingvalues.npy')):
            mixingvalues = np.load(os.path.expanduser('~/python/project/filestoload/Mixingvalues.npy'))
            # checks if energy/mixing values file exists to avoid reloading
        else:
            d = stardb.load(dbfilename)
            energyvalues = np.unique(d.fielddata['energy'])
            mixingvalues = np.unique(d.fielddata['mixing'])
                                   
            np.save(os.path.expanduser('~/python/project/filestoload/Energyvalues.npy'), energyvalues)
            np.save(os.path.expanduser('~/python/project/filestoload/Mixingvalues.npy'), mixingvalues)
            
        self.i = 0 # initialise the checker for data
        
        if (not reload) and os.path.isfile(os.path.expanduser('~/python/project/filestoload/IsoData.npy')):
            self.isodata = np.load(os.path.expanduser('~/python/project/filestoload/IsoData.npy'))
            self.remnantmasses = np.load(os.path.expanduser('~/python/project/filestoload/SpecificRem.npy'))
            self.isotopes = np.load(os.path.expanduser('~/python/project/filestoload/SpecificIso.npy'))
            self.massnumber = np.load(os.path.expanduser('~/python/project/filestoload/Massnumber.npy'))
            self.data = np.load(os.path.expanduser('~/python/project/filestoload/ProjectData.npy'))
            self.isoinfo = np.load(os.path.expanduser('~/python/project/filestoload/Ioninfo.npy'))
            self.time = self.data[1] 
            self.masses = self.data[0]
            self.explodemass = self.data[2]
            self.massfrac = self.data[3]
            self.egy = np.load(os.path.expanduser('~/python/project/filestoload/egy.npy'))
        else:
            self.data = np.array(makedata.data(dbfilename))
            self.i += 1  # need to run romberg and save data as data doesn't exist
            
            self.isodata = np.load(os.path.expanduser('~/python/project/filestoload/IsoData.npy'))
            self.isoinfo = np.load(os.path.expanduser('~/python/project/filestoload/Ioninfo.npy'))
            
            self.massnumber = np.load(os.path.expanduser('~/python/project/filestoload/Massnumber.npy'))
            self.time = self.data[1]
            self.masses = self.data[0]
            rem = np.load(os.path.expanduser('~/python/project/filestoload/RemnantMasses.npy'))
            
            self.remnantmasses = []
            rem = np.load(os.path.expanduser('~/python/project/filestoload/RemnantMasses.npy'))
            
            self.isotopes = []
            self.egy = []
            
            
            for starcount in range(len(self.masses)): # loop through each star individually

                s = str(self.masses[starcount])
                if s.endswith('.0'):                    # formatting issue, to match the filenames
                    s = s[:-2]                            
                filename = os.path.expanduser('~/python/project/dumps/z{}#presn').format(s)
                # grabs filename corrosponding to this mass
                d = kepdump.load(filename)
                energy = efunc(d)
                
                self.egy.append(energy)
                mixing = mixfunc(energy)
                remnantmass = remnantobtain.remnant(energyvalues, mixingvalues, energy, mixing,
                                                    rem, starcount)
                # this interpolates to grab the correct remnant mass
                
                isoarray = []
                for isotopecount in range(self.isodata.shape[2]):
                    k = elementobtain.element(energyvalues, mixingvalues, energy, mixing,
                                              self.isodata, starcount, isotopecount)
                    # interpolates to find the correct isotope ejecta
                    isoarray.append(k)
                if energy == 0.0: # this includes the start but no explosion/ejecta
                    remnantmass=self.masses[starcount]
                    isoarray=np.zeros(np.array(isoarray).shape)
                    
                self.isotopes.append(isoarray) 
                self.remnantmasses.append(remnantmass)
                
            reshape = np.array(self.isotopes)
            self.isotopes = np.swapaxes(reshape, 0, 1)    
            self.remnantmasses = np.array(self.remnantmasses)
            
            self.explodemass = self.masses-self.remnantmasses
            self.massfrac = self.explodemass/self.masses
            np.save(os.path.expanduser('~/python/project/filestoload/SpecificRem'), self.remnantmasses)
            np.save(os.path.expanduser('~/python/project/filestoload/SpecificIso'), self.isotopes)
            np.save(os.path.expanduser('~/python/project/filestoload/egy'), self.egy)
            np.save(os.path.expanduser('~/python/project/filestoload/oldimf'), 0)
        print('Please run the enterIMF(IMF=...) function. Salpeter is the default IMF')
                
    def enterIMF(self, IMF=salpeter.initialmassfunc):
        """
        Users enter desired IMF here by defining in the command line

        def yourimf(m):
            return m**2.35  or other function

        and use enterIMF(IMF=yourimf)
        """
        oldimf = np.load(os.path.expanduser('~/python/project/filestoload/oldimf.npy'))
        np.save(os.path.expanduser('~/python/project/filestoload/oldimf'), IMF)
        if oldimf == IMF:
            
            self.RBsol = self.data[5]
            self.isosol = np.load(os.path.expanduser('~/python/project/filestoload/Isosol.npy'))
            self.mastersum = np.load(os.path.expanduser('~/python/project/filestoload/Isocumsum.npy'))
            self.isointervalsum = np.load(os.path.expanduser('~/python/project/filestoload/Isointsum.npy'))
            self.numsteps = self.mastersum.shape[1]
            self.total = np.load(os.path.expanduser('~/python/project/filestoload/Totalsum.npy'))
        else:
            self.i += 1
            
            
###################################################################3

        # c found using 1 solar mass and integrating the IMF from min mass to max mass.

        
        self.IMF = lambda m: m*IMF(m) 
        self.c = 1/romberg(self.IMF, self.masses[0], self.masses[-1])

#####################################################################      
        # Interpolation of mass and time

        self.interpolationfunction = polyinterp.interp(3, self.time, self.masses)

#####################################################################   
        # Correcting the times for the interpolation function

        [self.correctedtime, self.deriv] = fixtime.fix(self.interpolationfunction, self.masses,
                                                       self.time)
              
#######################################################################
        #Linear interpolation of the mass fractions section
        
        self.fracinterp = interp1d(self.correctedtime, self.massfrac)
        self.isofracinterp = []
        
        for x in range(self.isotopes.shape[0]):
            interp = interp1d(self.correctedtime,self.isotopes[x]*self.massfrac*self.massnumber[x])
             
            self.isofracinterp.append(interp)
        self.isofracinterp = np.array(self.isofracinterp)
       
        
#######################################################################

        # THIS SECTION ONLY DONE IF IMF CHANGED
        
        if self.i > 0: # this meant the data was wrong or didn't exist
            [self.RBsol, self.isosol, self.mastersum, self.total, self.isointervalsum] = solutionsolve.solver(self, self.correctedtime, self.fracinterp, self.isofracinterp,
                                 self.interpolationfunction, self.c, self.isotopes, self.IMF)
            self.numsteps = self.mastersum.shape[1]
            np.save(os.path.expanduser('~/python/project/filestoload/Totalsum'), self.total)
            np.save(os.path.expanduser('~/python/project/filestoload/Isocumsum'), self.mastersum)
            np.save(os.path.expanduser('~/python/project/filestoload/Isointsum'), self.isointervalsum)

            # this willsave the initial mass/time data and its integration result
            txtdata = np.c_[self.masses, self.time, self.explodemass, self.massfrac, self.correctedtime, self.RBsol] # this will save the data into columns in a text file for easy reading
            
            with open(os.path.expanduser('~/python/project/outputfiles/initialdata.txt'), 'wt') as f:
                f.write(('{:>20s}'*(6) + '\n').format('Initial Mass', 'Lifetime', 'Explosion Mass', 'Mass Fraction', 'Corrected Time', 'Interval Solution'))
                for line in txtdata:
                    f.write(('{:>20.12e}'*(6) + '\n').format(*line.tolist()))

                    
            np.save(os.path.expanduser('~/python/project/filestoload/ProjectData'), [self.masses, self.time, self.explodemass, self.massfrac, self.correctedtime, self.RBsol])

            #this updates self.data to include all 6 fields
            self.data = np.load(os.path.expanduser('~/python/project/filestoload/ProjectData.npy'))
            np.save(os.path.expanduser('~/python/project/filestoload/Isosol'), self.isosol)
            self.i = 0
        
###################################################################

        self.datalabels = ('Initial Mass', 'Lifetime', 'Explosion Mass', 'Mass Fraction', 'Corrected Time', 'Interval Solution')
        
        self.tmin = self.correctedtime[-1]
        self.tmax = self.correctedtime[0]

################################################################### 
    # this is the master plotting code
    def plots(self):
        """
        plots is a module which can plot: mass/time, interpolation function, mass/correctedtime, massfrac/time, the function which is integrated.
        
        """
        
        self.plottingtool = projectplots.Plots(self.masses, self.time, self.interpolationfunction,
                                               self.correctedtime, self.fracinterp, self.massfrac,
                                               self.c, self.egy,self.IMF)
            

##################################################################            
    # This function gives the total mass yield out as a fraction of initial mass for a given time period        
    def massyield(self, tmin, tmax):

        """

        yieldcode will return total ejecta massfraction over a given time interval
        input t1 and t2. tmin=3.01e6 years, tmax=2e7 years.

        """

        
        self.intsum = yieldout.yieldcode(tmin, tmax, self.correctedtime, self.RBsol,
                                         self.fracinterp, self.interpolationfunction,
                                         self.c, self.IMF)        
        return self.intsum   

 
##########################################################################            
        #Plotting total
    def massyieldplot(self):
        """

        yieldplot will plot the total yield (massfraction of initial mass) over time
        """
        
        massyieldplotter.yieldplot(self.tmin, self.tmax, self.correctedtime, self.RBsol,
                                   self.fracinterp, self.interpolationfunction, self.c, self.IMF)

########################################################################        
        #this function will plot the yield output over time for desired isotopes
    def isotopeyieldplot(self, ions=None):

        """
        isotopeyieldplot plots the yield of the entered isotopes on a logscale alongwith the grand total of all isotopes ejected.

        Input desired isotopes as a string or isotope.ion array
        """
        isoyieldplt.iyplot(self.tmin, self.tmax, self.numsteps, self.mastersum,
                           self.isoinfo, self.total, ions)
       
############################################################################        
            
    # this code will take time interval input along with isotopes and it will calculate the yield of those isotopes over the desired time interval

    def isoyield(self, tmin, tmax, ions=None):
        """
        isoyield acts like massyield except it calculates the yield for
        any given isotope and time interval. 
        Input tmin, tmax, isotopes as a string or isotope.ion array

        """
        topes = ions
        if topes is None:
            topes = input('Enter desired isotopes: ')
        
        if isinstance (topes, str):
            topes = re.split('[ ,;]+', topes)
        topes = np.array([isotope.ion(x) for x in topes])
        
        
        for x in range(len(topes)):
            topeindex = np.where(topes[x]==self.isoinfo)[0][0]

            self.intsum = yieldout.yieldcode(tmin, tmax, self.correctedtime, self.isosol[topeindex], self.isofracinterp[topeindex], self.interpolationfunction,  self.c, self.IMF)
            
            print('The yield for ', self.isoinfo[topeindex],' is ', self.intsum)
          
#############################################################################

    # code here to produce table of values every ~100,000 years  for each interval

    def datatable(self):
        """
        Produces a table of values every ~100,000 years of the ejected mass fraction for every isotope from that period.
	
        """	
        numsteps = self.isointervalsum.shape[1]
        tinterval = (self.tmax-self.tmin)/numsteps
        table = []
        t = 0
        ti = []
        
        for x in range(numsteps):
            t += tinterval
            ti.append(t)
        table.append(ti)
        elnames = []
        
        for x in range(len(self.isoinfo)):
            table.append(self.isointervalsum[x])
            elnames.append(self.isoinfo[x])
        self.table = np.array(table)
        reshape = np.swapaxes(self.table, 0, 1)

        nel = len(self.isoinfo)
        
        with open(os.path.expanduser('~/python/project/outputfiles/noexplisoyieldtable.txt'), 'wt') as f:
            f.write(('{:>20s}'*(1+nel) + '\n').format('time', *elnames))
            for line in reshape:
                f.write(('{:>20.12e}'*(1+nel) + '\n').format(*line.tolist()))

##########################################################
    # will produce a stack plot of contribution by isotopes
    def stackplot(self, ions = None):
        """
        Produces a plot of contributions of each isotope towards the total ejected fraction
	also plots the rate of ejection over time for each isotope.

        Input desired isotopes as a string or isotope.ion vector
        """
        topes = ions
        if topes is None:
            topes = input('Enter desired isotopes: ')
        
        if isinstance (topes, str):
            topes = topes.split()
        topes = np.array([isotope.ion(x) for x in topes])
            
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        taxis = np.linspace(self.tmin, self.tmax, self.numsteps)
        y = []

        # this will collect the required isotopes to be stacked, will also create the legend as stackplot can't do it.
        
        for x in range(len(topes)):
            topeindex = np.where(topes[x]==self.isoinfo)[0][0]
            y.append(self.mastersum[topeindex])
            ax.plot([],[],label=self.isoinfo[topeindex].LaTeX())
        ax.plot(taxis, self.total, '--', label='Grand Total')
        
        # this resets the colour cycle to default so the stackplot colours match with the legend, generalises the code and allowsthe user to select any number of isotopes    
        plt.gca().set_prop_cycle(None)
        
        ax.stackplot(taxis, y)
        ax.set_ylabel('Ejecta Mass Fraction')
        ax.set_xlabel('Time(years)')
        ax.set_title('Contribution by isotopes to ejecta.')
        ax.legend(loc='upper left')
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.expanduser('~/python/project/outputfiles/stackplot.pdf'), bbox_inches='tight')

        fig = plt.figure(2)
        plt.gca().set_prop_cycle(None)
        bx = fig.add_subplot(1,1,1)
        for x in range(len(topes)):
            topeindex = np.where(topes[x]==self.isoinfo)[0][0]
            m= lambda t: np.exp(self.interpolationfunction(np.log(t)))
            y= lambda t: -self.IMF(m(t))*self.c*self.deriv(np.log(t))*self.isofracinterp[topeindex](t)/t*m(t)
            bx.plot(taxis,y(taxis), label=self.isoinfo[topeindex].LaTeX())
        bx.set_ylabel('Ejecta Rate (massfraction per year)')
        bx.set_xlabel('Time(years)')
        bx.set_title('Rate of mass ejection over time.')
        bx.legend(loc='upper right')
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.expanduser('~/python/project/outputfiles/ejectarate.pdf'), bbox_inches='tight')
