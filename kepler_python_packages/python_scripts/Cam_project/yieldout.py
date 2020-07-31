import numpy as np
from scipy.integrate import romberg


# Main code for finding yield            
def yieldcode(tmin, tmax, time, RBsol, fracinterp, polynomial, c, IMF):
    """
    The yieldcode module will use the step solution from the rombergquad
    module to calculate the yield in any time period, by calculating and
    excess outside the step ranges.
    """

    if tmin < time[-1]:
        tmin = time[-1]
        
    if tmax > time[0]:
        tmax = time[0]

    deriv = polynomial.deriv()    

    lowerindex = np.where(time<=tmax)[0][0]
    upperindex = np.where(time>=tmin)[0][-1]
   

    if lowerindex >= upperindex: # this is to avoid counting romberg intervals when the times don't actually cover any interval.
        m = lambda t: np.exp(polynomial(np.log(t)))
        y = lambda t: IMF(m(t))*c*deriv(np.log(t))*fracinterp(t)/t*m(t)    
        intsum = romberg(y, tmax, tmin)
        
    else:
        intervals = RBsol[lowerindex:upperindex] 
    
        intsum = np.sum(intervals)
    
        
        # Finding the excess integrals
        m = lambda t: np.exp(polynomial(np.log(t)))
        y = lambda t: IMF(m(t))*c*deriv(np.log(t))*fracinterp(t)/t*m(t) 
       
        t1 = time[lowerindex]   # corrected time is a decreasing vector
        t2 = tmax

        sol = romberg(y, t2, t1)
        intsum += sol
        
        
        t1 = tmin   
        t2 = time[upperindex]

        sol = romberg(y, t2, t1)
        intsum += sol
        
    return intsum
