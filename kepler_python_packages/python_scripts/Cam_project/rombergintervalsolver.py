import numpy as np    
from scipy.integrate import romberg, quad


def rombergquad(time, fracinterp, polynomial, c, IMF):
    """
    The rombergquad module is the solver which creates the big
    function to be interpolated and then solves it numerically using
    the adaptive step size romberg method and stores the solution in
    sections for each of the steps in the piecewise fraction
    function
    """
    deriv = polynomial.deriv()
    
    m = lambda t: np.exp(polynomial(np.log(t)))
    y = lambda t: IMF(m(t))*c*deriv(np.log(t))*fracinterp(t)/t*m(t)    
    numberofinterval = len(time)

    rombergsol = []
    for n in range(1,numberofinterval):
            
        t1 = time[n]   # corrected time is a decreasing vector, t2>t1
        t2 = time[n - 1]
        sol = romberg(y, t2, t1)
        rombergsol.append(sol)
            
            
    rombergsol.append(0) # to make the vector the same length as the others for data storage as it is the intevals so one less value than data points in time or mass.
    RBsol = np.array(rombergsol) 
    return RBsol
