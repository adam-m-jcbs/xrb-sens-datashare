        #Fixing the times section
import numpy as np    

def fix(poly, masses, time):
    """
    Fix will use the newton raphson rootfinding method to
    translate the data points to the interpolation function curve
    """
    newtime = []

    deriv = poly.deriv()
    index = 0
    for mass in masses:
           

        m = np.log(mass)    
        xguess = np.log(time[index]) # is the initial guess, will use the real time
        error = 1
        index += 1
        while error>1e-9:
            xguess = xguess - (poly(xguess) - m)/deriv(xguess) # using newton raphson method
                
            error = poly(xguess) - m
                    
        newtime.append(xguess)     
                    
    correctedtime = np.exp(np.array(newtime)) 
    return (correctedtime, deriv)

