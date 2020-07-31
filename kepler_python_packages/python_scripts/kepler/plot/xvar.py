import numpy as np

from ..datainterface import DataInterface

from physconst import Kepler as const

class XVar():
    vfac =  3 / (4 * np.pi)
    def __init__(self, data, irtype = None):
        assert isinstance(data, DataInterface)
        self.data = data
        self.irtype = irtype
        self.irtype_current = None
        self.update()

    def update(self, *args, **kwargs):
        irtype = kwargs.get('irtype', None)
        data = self.data
        if irtype is None:
            irtype = self.irtype
        if irtype is None:
            irtype = data.irtype
        jm = data.jm
        iib = slice(0, jm+1)
        iil = slice(0, jm)
        iim = slice(1, jm+1)
        iih = slice(1, jm+1)
        x = None
        xm = None
        scale = 'linear'
        name = None
        if irtype == 1:
            label = r'log Radius (cm)'
            x = data.rn[iib]
            if x[0] == 0:
                x[0] = 0.5 * x[1]
            x = np.log10(x)
            name = 'log rn'
        elif irtype == 2:
            label = r'Interior Mass Fraction'
            x = data.zm[iib]
            x = x - x[0]
            x = x / x[-1]
            name = 'q'
        elif irtype == 3:
            label = r'Interior Mass (solar masses)'
            x = data.zm[iib] / const.solmass
            name = 'zm_sun'
        elif irtype == 4:
            label = r'Radius (cm)'
            x = data.rn[iib]
            name = 'rn'
        elif irtype == 5:
            label = r'log Interior Mass (solar masses)'
            x = data.zm[iib] / const.solmass
            if x[0] == 0:
                x[0]=0.25*x[1]
            x = np.log10(x)
            name = 'log zm_sun'
        elif irtype == 6:
            label = r'Zone Number'
            x = np.arange(jm+2, dtype = np.float64)[iib]
            name = 'zone'
        elif irtype == 7:
            label = r'log Interior Mass (solar masses)'
            x = np.zeros(jm+1)
            x[iih] = np.cumsum(data.xm[iim]) / const.solmass
            x[0] = 0.5 * x[1]
            x = np.log10(x)
            name = 'log zm_sun grid'
        elif irtype == 8:
            label = r'Interior Mass (solar masses)'
            x = np.zeros(jm+1)
            x[iih] = np.cumsum(data.xm[iim]) / const.solmass
            name = 'zm_sun grid'
        elif irtype == 9:
            label = r'log Exterior Mass (solar masses)'
            x = np.zeros(jm+1)
            x[iil] = np.cumsum(data.xm[iim][::-1])[::-1]
            if data.xmacrete > 0:
                x += data.xmacrete
            else:
                x[jm] = 0.5 * x[jm-1]
            x /= const.solmass
            x = np.log10(x)
            name = 'log ym_sun'
        elif irtype == 10:
            label = r'Exterior Mass (solar masses)'
            x = np.zeros(jm+1)
            x[iil] = np.cumsum(data.xm[iim][::-1])[::-1]
            x += data.xmacrete
            x /= const.solmass
            name = 'ym_sun'
        elif irtype == 11:
            label = r'log Column Density ($\mathrm{g}\,\mathrm{cm}^{-2}$)'
            x = data.y
            if x[jm] == 0:
                x[jm] = 0.5 * x[jm-1]
            x = np.log10(x)
            name = 'y'
        elif irtype == 12:
            label = r'Column Density ($\mathrm{g}\,\mathrm{cm}^{-2}$)'
            x = data.y
            name = 'log y'
        elif irtype == 13:
            label = r'Pressure ($\mathrm{erg}\,\mathrm{cm}^{-3}$)'
            xm = data.pn[iib]
            x = data.pnf[iib]
            name = 'pn'
        elif irtype == 14:
            label = r'log Pressure ($\mathrm{erg}\,\mathrm{cm}^{-3}$)'
            xm = data.pn[iib]
            xm[0] = xm[1]
            xm = np.log10(xm)
            x = np.log10(data.pnf[iib])
            name = 'log pn'
        elif irtype == 15:
            label = r'Gravitational Potential ($\mathrm{cm}^2\,\mathrm{s}^{-2}$)'
            x = -data.phi[iib]
            name = 'phi'
        elif irtype == 16:
            label = r'log Gravitational Potential ($\mathrm{cm}^2\,\mathrm{s}^{-2}$)'
            x = np.log10(-data.phi[iib])
            name = 'log phi'
        elif irtype == 17:
            label = r'Gravitational Potential ($\mathrm{c}^2$)'
            x = -data.phi[iib] / const.c**2
            name = 'phi/c**2'
        elif irtype == 18:
            label = r'log Gravitational Potential ($\mathrm{c}^2$)'
            x = np.log10(-data.phi[iib] / const.c**2)
            name = 'log phi/c**2'
        elif irtype == 19:
            label = r'redshift'
            x = -data.phi[iib]
            x = x / (const.c**2 - x)
            name = 'z'
        elif irtype == 20:
            label = r'log redsshift'
            x = -data.phi[iib]
            x = x / (const.c**2 - x)
            x = np.log10(x)
            name = 'log z'
        elif irtype == 21:
            label = r'Enclosed Volune ($\mathrm{cm}^3$)'
            x = data.rn[iib]**3 * self.vfac
            name = 'vol'
        elif irtype == 22:
            label = r'log Enclosed Volune ($\mathrm{cm}^3$)'
            x = data.rn[iib]**3 * self.vfac
            if x[0] == 0:
                x[0] = 0.125*x[1]
            x = np.log10(x)
            name = 'log vol'
        elif irtype == 23:
            label = r'Enclosed Volune ($\mathrm{V}_\odot$)'
            x = (data.rn[iib] / const.solrad)**3
            name = 'vol_sun'
        elif irtype == 24:
            label = r'log Enclosed Volune ($\mathrm{V}_\odot$)'
            x = (data.rn[iib] / const.solrad)**3
            if x[0] == 0:
                x[0] = 0.125*x[1]
            x = np.log10(x)
            name = 'log vol_sun'
        elif irtype == 25:
            label = r'optical depth'
            x = data.tau[iib]
            name = 'tau'
        elif irtype == 26:
            label = r'log optical depth'
            x = data.tau[iib]
            if x[-1] == 0:
                x[-1] = 0.25 * x[-2]
            x = np.log10(x)
            name = 'log tau'
        elif irtype == 27:
            label = r'Moment of Inertia ($\mathrm{g}\,\mathrm{cm}^2$)'
            x = data.angit[iib]
            name = 'angi'
        elif irtype == 28:
            label = r'log Moment of Inertia ($\mathrm{g}\,\mathrm{cm}^2$)'
            x = data.angit[iib]
            if x[0] == 0:
                x[0] = 0.25 * x[1]
            x = np.log10(x)
            name = 'log angi'
        elif irtype == 29:
            label = r'Moment of Inertia ($\mathrm{M}_\odot\,\mathrm{R}_\odot^2$)'
            x = data.angit[iib] / (const.solmass * const.solrad**2)
            name = 'angi_sun'
        elif irtype == 30:
            label = r'log Moment of Inertia ($\mathrm{M}_\odot\,\mathrm{R}_\odot^2$)'
            x = data.angit[iib] / (const.solmass * const.solrad**2)
            if x[0] == 0:
                x[0] = 0.25 * x[1]
            x = np.log10(x)
            name = 'log angi_sun'
        else:
            raise Exception('['+self.__class__.__name__+'] irtype "{!r}" not supported.'.format(irtype))
        if xm is None:
            xm = np.array(
                [x[0]] +
                (0.5*(x[iil] + x[iih])).tolist() +
                [x[jm]])
        if x is None:
            # get boundaries and centers by linerar inter- and extrapolation
            x = np.array(
                [1.5*xm[1] - 0.5*xm[2]] +
                (0.5*(xm[1:jm] + xm[2:jm+1])).tolist() +
                [1.5*xm[jm] - 0.5*xm[jm-1]])

        self.scale = scale
        self.irtype_current = irtype
        self.label = label
        self.name = name
        self.xm = xm
        self.x = x
        self.jm = jm
        self.iib = iib
        self.iil = iil
        self.iim = iim
        self.iih = iih
