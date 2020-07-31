import numpy as np

        

def interp(power, time, masses):
    """
    This module creates the polynomial interpolation function for the mass/time data
    """
    
    logtime = np.log(time)
    logmasses = np.log(masses)
    coefs = np.polyfit(logtime,logmasses, power)

    polynomial = np.poly1d(coefs) # creates a polynomial function
    

    return polynomial
