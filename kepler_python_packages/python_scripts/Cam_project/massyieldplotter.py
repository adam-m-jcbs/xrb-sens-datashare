import numpy as np
import matplotlib.pyplot as plt
import yieldout, stardb, os
def yieldplot(tmin, tmax, correctedtime, RBsol, fracinterp, interpolationfunction, c, IMF):
    """Plots the total mass yield out as a fraction of initial mass over time"""    
    numsteps = 170
    # roughly 100,000yrs
    stepsize = ((tmax - tmin)/numsteps)
    y = []

        # loop through and solve each intervals yieldout for cumulative summation
    for x in range(numsteps):       
            

        sol = yieldout.yieldcode(tmin + x*stepsize, tmin + (x + 1)*stepsize,
                                 correctedtime, RBsol, fracinterp, interpolationfunction,  c, IMF)
            # stores each interval's solution
        y.append(sol)
            
        # cumulative sums each interval    
        fracsum = np.cumsum(y)
        
    taxis = np.linspace(tmin, tmax, len(fracsum))
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.plot(taxis, fracsum)
    ax.set_ylabel('Mass fraction ejected')
    ax.set_xlabel('Time(years)')
    ax.set_title('Mass ejection over time')
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.expanduser('~/python/project/outputfiles/massyieldplot.pdf'), bbox_inches = 'tight')
