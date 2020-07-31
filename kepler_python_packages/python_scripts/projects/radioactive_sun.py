# /bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

I=[
    [1.25,0.0000000621425229175332,1.003110198],
    [1.8,0.0000055388716539354100,1.925034877],
    [3,0.0002620000000000000000,2.76876],
    [4,0.0002814690000000000000,1.670435563],
    [5,0.0000013911014237458600,0.992366956],
    [6,0.0000015340985031590100,0.993398682],
    [6.5,0.0000026496787948853900,0.986285093],
    [8.,0.000001784,0.99733256],
    [8.5,0.00000570,0.97409],
    [12,0.0097833600000000000000,0.915914076],
    [15,0.0144417070000000000000,0.871334054],
    [18,0.0255048090000000000000,0.825512138],
    [25,0.0230063840000000000000,0.761492641]]

Hf=[
    [1.25,0.010181722,1.02157355],
    [1.8,0.019937662,17.23735471],
    [3,0.149028,27.92228],
    [4,0.281106178,10.55675685],
    [5,0.017511563,1.033507569],
    [6,0.013277734,1.025960932],
    [6.5,0.008186895,1.02383545],
    [8.,0.00526,1.03293977],
    [8.5,0.00397,1.016828],
    [12,0.047984583,1.012192068],
    [15,0.046354969,1.021806267],
    [18,0.144685666,1.107324388],
    [25,0.193241478,1.03707803]]

Pd=[
    [1.25,0.050207867,1.420802496],
    [1.8,0.14285228,11.96525037],
    [3,0.142159,24.986],
    [4,0.130211833,8.42171815],
    [5,0.002060577,1.004621526],
    [6,0.002360262,1.004750483],
    [6.5,0.009049768,1.020787672],
    [8.,0.0081371,1.02353],
    [8.5,0.0144,1.03105],
    [12,0.024802885,0.985315835],
    [15,0.038937327,0.958736493],
    [18,0.034673491,0.957441191],
    [25,0.054011734,1.024450849]]

data = np.array([Pd,I,Hf])

gap = 9
slices = [np.s_[:gap],np.s_[gap:]]

def plot():
    f = plt.figure(
            figsize = (6,8),
            dpi = 102,
            facecolor = 'white',
            edgecolor = 'white'
            )
    f.subplots_adjust(hspace=1.e-10)

    colors = ['r','g','b']
    markers = ['s','o','^']
    labels = np.array([[
        r'$^{107\!}\mathrm{Pd}$ / $^{108\!}\mathrm{Pd}$',
        r'$\;^{129\!}\mathrm{I}$ / $^{127\!}\mathrm{I}$',
        r'$^{182\!}\mathrm{Hf}$ / $^{180\!}\mathrm{Hf}$',
        ],[
        r'$^{108\!}\mathrm{Pd}$ / $^{108\!}\mathrm{Pd}_\odot$',
        r'$\;^{127\!}\mathrm{I}$ / $^{127\!}\mathrm{I}_\odot$',
        r'$^{180\!}\mathrm{Hf}$ / $^{180\!}\mathrm{Hf}_\odot$',
        ]])

    ax1 = f.add_subplot(211)
    ax1.set_xlim(1,30)
    ax1.set_xscale('log')
    ax1.set_ylim(2.e-8,1)
    ax1.set_yscale('log')
    ax1.set_ylabel('Ratio by Number')

    ax2 = f.add_subplot(212, sharex=ax1)
    ax2.set_xlim(1,30)
    ax2.set_ylim(.5,40)
    ax2.set_yscale('log')
    ax2.set_ylabel('Production Factor')
    ax2.set_xlabel(r'Initial Stellar Mass / Solar Masses')

    ax = [ax1, ax2]
    panels = ['A','B']

    for j,a in enumerate(ax):
        for i in range(3):
            l = labels[j,i]
            for s in slices:
                a.plot(data[i,s,0], data[i,s,j+1],
                       linestyle = '-',
                       marker = markers[i],
                       color = colors[i],
                       label = l)
                l = None
        a.legend(loc = 'best', ncol=1, fontsize=12)
        a.text(.01, 0.98, panels[j],
                 horizontalalignment = 'left',
                 verticalalignment = 'top',
                 size = 15,
                 weight = 'semibold',
                 color = 'k',
                 transform = a.transAxes,
                 )

    for x in ax1.get_xticklabels():
        x.set_visible(False)
    ax2.set_xticklabels(['','1','10'])
    ax2.set_yticklabels(['','','1','10'])


    f.tight_layout()
    f.subplots_adjust(hspace=1.e-10)
    plt.show()

if __name__ == '__main__':
    plot()
