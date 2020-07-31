import numpy as np
import rombergintervalsolver, yieldout
def solver(self, correctedtime, fracinterp, isofracinterp, interpolationfunction, c, isotopes, IMF):

    """
    This module solves uses the yieldcode above and the
    rombergquad to obtain the solutions for both the mass abd
    isotopes and create solution arrays and cumulative solution
    arrays over the whole time period
    """
    self.correctedtime = correctedtime
    self.fracinterp = fracinterp
    self.isofracinterp = isofracinterp
    self.interpolationfunction = interpolationfunction
    self.c = c
    self.isotopes = isotopes
    self.IMF = IMF
    self.RBsol = rombergintervalsolver.rombergquad(self.correctedtime, self.fracinterp,
                                                   self.interpolationfunction,  self.c, self.IMF)


    self.isosol = []
    for x in range(self.isotopes.shape[0]):
            
        sol = rombergintervalsolver.rombergquad(self.correctedtime, self.isofracinterp[x],
                                                self.interpolationfunction,  self.c, self.IMF)
        self.isosol.append(sol)
             
        
    self.isosol = np.array(self.isosol)

    ###############################
    # for calculating cumulative sum of every isotope
    self.numsteps = 170
    # roughly 100,000yrs
    self.tmin = self.correctedtime[-1]
    self.tmax = self.correctedtime[0]
    stepsize = ((self.tmax - self.tmin)/self.numsteps)
    self.mastersum = []
    self.isointervalsum = []
    # have to loop the solve/plot for each isotope
    for q in range(self.isotopes.shape[0]):
        y = []

        # this loop does the cumulative sum over time for a single isotope
        for x in range(self.numsteps):       
            
            sol = yieldout.yieldcode(self.tmin+x*stepsize, self.tmin+(x+1)*stepsize,
                                     self.correctedtime, self.isosol[q], self.isofracinterp[q],
                                     self.interpolationfunction,  self.c, self.IMF)
            # stores each interval's solution
            y.append(sol)
            
        # cumulative sums each interval    
        fracsum = np.cumsum(y)
        self.mastersum.append(fracsum)
        self.isointervalsum.append(y)
        
    self.total = []
        
    for y in range(self.numsteps):
        sums = 0           
        for x in range(self.isotopes.shape[0]):
            sums += self.mastersum[x][y]
        self.total.append(sums)
    self.total = np.array(self.total)            
    self.mastersum = np.array(self.mastersum)
    self.isointervalsum = np.array(self.isointervalsum)
    # mastersum is an array of each isotope's cumulative yield value at each of the 170 steps
    # total is a vector of the total isotope yield at each step
    # isointervalsum is the individual interval solutions
    
    return (self.RBsol, self.isosol, self.mastersum, self.total, self.isointervalsum)
