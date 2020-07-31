import numpy as np
import matplotlib.pyplot as plt
import isotope, re, os

def iyplot(tmin, tmax, numsteps, mastersum, isoinfo, total, ions):
 
    """
    Isotopeyieldplot plots the yield of the entered isotopes on a logscale alongwith the grand total of all isotopes ejected.
    """

    topes = ions
    if topes is None:
        topes = input('Enter desired isotopes: ')
    
    if isinstance (topes, str):
        topes = re.split('[ ,;]+', topes)
    topes  =  np.array([isotope.ion(x) for x in topes])
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    taxis = np.linspace(tmin, tmax, numsteps)
        
    for x in range(len(topes)):
        topeindex = np.where(topes[x] == isoinfo)[0][0]
        ax.plot(taxis, mastersum[topeindex], label = isoinfo[topeindex].LaTeX())
            

    ax.plot(taxis, total, '--', label = 'Grand Total')
    ax.set_ylabel('Ejecta Mass Fraction')

    ax.set_xlabel('Time(years)')
    ax.set_yscale('log')
        
    ax.set_title('Mass ejection over time for star dependent energy')
    ax.legend(loc = 'best')
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.expanduser('~/python/project/outputfiles/noexplisoyieldplot.pdf'), bbox_inches = 'tight')
