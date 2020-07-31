
# matplotlib.use("module://mplcairo.xt")
import numpy as np
from numpy import mgrid
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os.path
import itertools

try:
    from mpl_toolkits.basemap import Basemap
except:
    print('Could not import basemap.')

from rotation import rotscale, rotate2, w2p, p2l

from movie import MovieWriter
from movie.mplbase import MPLBase
from movie import ParallelMovieFrames, ParallelMovie, PoolMovie

from matplotlib.scale import SymmetricalLogScale
from matplotlib.cm import get_cmap
from matplotlib import rcParams


from color import ColorBlend, ColorBlendWave

from human import time2human

pi = np.pi
sin = np.sin
cos = np.cos
arccos = np.arccos
arctan = np.arctan
arctan2 = np.arctan2
sqrt = np.sqrt
r2d = 180/pi
log = np.log

from scipy.special import erfinv
from scipy.special import erf


def hat(theta, phi):
    s = sin(theta)
    x = s * cos(phi)
    y = s * sin(phi)
    z = cos(theta)
    return np.array([x,y,z])


class DSIPlot(object):
    def __init__(
            self, *,
            align = None,
            j = [0,0,1],
            dj = [1,0,0],
            w = None,
            dw = None,
            n = 2**10,
            mode = 'shade',
            fig = None,
            ax = None,
            alignphi = 0,
            proj = 'ortho',
            n2 = 1,
            k = 2/3, # gyration constant j = w * k * r **2
            ra = 1, # radius
            fsh = 1, # enable N2W term; set to 0 for 'classical' DSI
            ric = 1/4,
            aspect = None,
            vmin = None,
            vmax = None,
            mag = None,
            cbo = None,
            cmap = 'bwr_r',
            calc_mode = 'min', # min | max
            ):

        theta, phi = mgrid[0:pi:(n+1)*1j, 0:2*pi:(n*2+1)*1j]
        r_hat = hat(theta, phi)
        r = ra * r_hat

        i = k * ra ** 2

        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            j = w * i
        else:
            j = np.asarray(j, dtype=np.float64)
            w = j / i
        # dw = dw/dr
        # dj = dj/dr
        if dw is not None:
            dw = np.asarray(dw, dtype=np.float64)
            dj = dw * i + 2 * j / ra
        else:
            dj = np.asarray(dj, dtype=np.float64)
            dw = dj / i - 2 * w / ra

        note = """
        dlnw/dlnr
        = dw * ra / w
        = dj * ra / (i * w )  - 2 w * ra / (ra * w)
        = dj * ra / j - 2
        = dlnj/dlnr - 2

        and (hence, obviously,)
        dlnj/dlnr =
        = dj * ra / j
        = dw * i * ra / j + 2 * j * ra / (ra * j)
        = dw * ra / w + 2
        dlnw/dlnr + 2

        mud
        = dw . dj / (|dw|.|dj|)
        = (dj / i - 2 * w / ra) * dj
        = (dj . dj / i - 2 * w * dj / ra) / (|dw| * |dj|)
        = |dj| / (i * |dw|) - 2 ...
        """

        if align is None:
            if mode == 'sphere':
                align = False
            else:
                align = True

        muj = np.dot( j, dj) / (norm(j)  * norm(dj))
        muw = np.dot( w, dw) / (norm(w)  * norm(dw))
        mud = np.dot(dw, dj) / (norm(dj) * norm(dw))

        # align such that j || z and dj in x-z plane
        if align:
            j = np.array([0, 0, 1.]) * norm(j)
            w = np.array([0, 0, 1.]) * norm(w)
            dj = np.array([sqrt(1-muj**2)*cos(alignphi),
                           sqrt(1-muj**2)*sin(alignphi),
                           muj]) * norm(dj)
            dw = dj / i - 2 * w / ra

        # vector
        jdotrhat = np.tensordot( j, r_hat, (0,0))
        djdotrhat = np.tensordot(dj, r_hat, (0,0))
        jdotdj = np.dot(j, dj)

        mu = cos(theta)
        p = phi/(2*pi)

        nw1 = jdotrhat * djdotrhat
        nw2 = jdotdj
        n2w = 2 * (nw2 - nw1) / (ra**3 * k**2) * fsh

        #DSI
        dwdlr = np.cross(dw, r, axis=0)
        dwdlr2 = np.sum(dwdlr * dwdlr, axis=0)

        # Ri = (N2 + N2W) / dwdr**2 or classical just N2 / dwdr**2
        # ri = (n2 + fsh * n2w) / dwdlr2

        n2r = - ric * dwdlr2

        # generalised criterion
        n2x = n2 + n2w + n2r

        # diagnostic output
        n2xp = np.max(n2x)
        n2xm = np.min(n2x)

        if mode == 'calc':
            n2xmag = n2xp - n2xm

            if calc_mode == 'min':
                opt = np.min
                arg = np.argmin
                self.n2xm = n2xm
            else:
                opt = np.max
                arg = np.argmax
                self.n2xm = n2xp

            self.n2m = opt(n2)
            self.n2rm = opt(n2r)
            self.n2wm = opt(n2w)

            ii = arg(n2x)
            jt = ii // n2x.shape[1]
            jp = ii - jt * n2x.shape[1]

            # some roundoff of not really known origin
            eps = 1.e-3
            if jp != 0:
                if calc_mode == 'min':
                    if np.abs(n2x[jt,jp] - np.min(n2x[:,0])) < eps * n2xmag:
                        jp = 0
                        jt = np.argmin(n2x[:, 0])
                    else:
                        print(np.abs(n2x[jt,jp] - np.min(n2x[:,0])), eps * n2xmag)
                else:
                    if np.abs(n2x[jt,jp] - np.max(n2x[:,0])) < eps * n2xmag:
                        jp = 0
                        jt = np.argmax(n2x[:, 0])
                    else:
                        print(np.abs(np.max(n2x[:,0] - n2x[jt,jp])), eps * n2xmag)

            self.n2xmt = theta[jt, jp]
            self.n2xmp = phi[jt, jp]

            self.n2xam = opt(np.average(n2x, axis=1))
            ii = arg(np.average(n2x, axis=1))
            self.n2xamt = theta[ii, 0]

            n2xtm = opt(n2x, axis=1)
            self.n2xtm = n2xtm
            self.t = theta[:,0]

            eps = 1.e-3
            if calc_mode == 'min':
                ii = np.where(n2xtm < n2xm + eps * n2xmag)[0]
            else:
                ii = np.where(n2xtm > n2xm - eps * n2xmag)[0]
            self.tmin = np.min(self.t[ii])
            self.tmax = np.max(self.t[ii])

            self.dw = dw
            self.dj = dj
            self.w = w
            self.j = j

            self.muw = muw
            self.muj = muj

            return

        print(f'cos( j, dj)={muj:5.3f}')
        print(f'cos( w, dw)={muw:5.3f}')
        print(f'cos(dw, dj)={mud:5.3f}')

        print(f'[LOCAL] N2x maximum = {n2xp:4.2f}, minimum = {n2xm:4.2f}')

        scosj = rf'\cos\left(\vec{{j}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right)={muj:+4.2f}'
        scosw = rf'\cos\left(\vec{{\omega}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{\omega}}\right)={muw:+4.2f}'
        scosd = rf'\cos\left(\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{\omega}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right)={mud:+4.2f}'

        sdj = ''
        if n2 != 0:
            sdj += rf'N^2'
        if fsh != 0:
            if sdj != '':
                sdj += '+'
            sdj += rf'N^2_{{\Omega}}'
        if ric != 0:
            sdj += rf'-Ri_\mathrm{{c}}\,\left(\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\vec{{\omega}}\times\vec{{r}}\right)^2'
        sdjn = rf'{sdj}'

        jpol=w2p(j)
        djpol=w2p(dj)
        wpol=w2p(w)
        dwpol=w2p(dw)
        jpolr=w2p(-j)
        djpolr=w2p(-dj)
        wpolr=w2p(-w)
        dwpolr=w2p(-dw)

        if fig is None:
            fig, ax = plt.subplots()
        if ax is None:
            ax = fig.add_subplot(111)

        sleg = r'unstable $\longleftrightarrow$ stable'

        if mag is None:
            mag = max(abs(n2xp), abs(n2xm))
        if mag == 0:
            mag = np.abs(n2) + 2 * norm(j)*norm(dj)/ra**3 + ric * norm(dw)**2*ra**2
        if vmin is None and mag is not None:
            vmin = -mag
        if vmax is None and mag is not None:
            vmax = +mag

        if mode in ('shade', 'sphere', ):
            n2xc = f2c(n2x)

        if mode == 'contour':
            cs = ax.contour(p, mu, n2x)
            ax.clabel(cs, inline=True, fontsize=8)
        elif mode == 'shade':
            cs = ax.pcolormesh(p, mu, n2xc, vmin=vmin, vmax=vmax, cmap=cmap)
        elif mode == 'sphere':
            # good are:
            #     ortho, moll, merc, cyl, gal, mill, hammer, nsper, eck4, kav7,
            #     vandg, robin
            # m = Basemap(projection=proj, ax=ax, lat_0=-37.907803,lon_0=145.133957)
            m = Basemap(projection='ortho', ax=ax, lat_0=-37.907803,lon_0=145.133957)
            m.drawmeridians(np.arange(0,360,30)) # grid every 30 deg
            m.drawparallels(np.arange(-90,90,30))
            m.drawcoastlines(color='#ffffff3f')
            lon = r2d * phi
            lon[lon > 180] -= 360
            lat = 90 - r2d * theta
            x, y = lon, lat
            if False:
                cs = m.pcolor(lon, lat, ri, vmin=vmax, vmax=vmin, cmap=cmap, latlon=True)
            else:
                # fix for pcolormesh (created pull request)
                n2xc = pcm_fix(*m(lon,lat),n2xc)
                # x,y = m(lon,lat)
                # nx,ny = x.shape
                # ii = (
                #     (x[:-1,:-1] > 1e20) |
                #     (x[1:,:-1] > 1e20) |
                #     (x[:-1,1:] > 1e20) |
                #     (x[1:,1:] > 1e20) |
                #     (y[:-1,:-1] > 1e20) |
                #     (y[1:,:-1] > 1e20) |
                #     (y[:-1,1:] > 1e20) |
                #     (y[1:,1:] > 1e20)
                #     )
                # n2xc[:nx-1,:ny-1][ii]=np.nan
                cs = m.pcolormesh(lon, lat, n2xc, vmin=vmin, vmax=vmax, cmap=cmap, latlon=True)
            leg_loc = 'upper right'

            stit = f'${sdjn}$'
            fig.text(0.01, 0.99, stit, va='top', fontsize='large', color='k')

        elif mode in ('project', 'min', ):
            if mode == 'min':
                n2xa = np.min(n2x, axis=1)
                sxleg = rf'$\min\left(\,{sdjn}\,\right)$'
            else:
                n2xa = np.average(n2x, axis=1)
                sxleg = rf'$\left<\,{sdjn}\,\right>$'
            ax.plot(n2xa, mu[:,0])

            n2xap = np.max(n2xa)
            n2xam = np.min(n2xa)

            print(f'[AVERAGES] maximum = {n2xap:4.2f}, minimum = {n2xam:4.2f}')

            leg_loc = 'upper right'
            ax.set_ylabel(r'$\mu$')
            ax.set_xlabel(sxleg)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(-1, 1)

        if mode in ('contour', 'shade', ):
            stit = fr'${sdjn}$'
            if fsh != 0:
                stit += fr'$,\quad{scosj}$'
            if ric != 0:
                stit += fr'$,\quad{scosw}$'
            ax.set_title(stit)

            ax.set_xlabel(r'Phase')
            ax.set_ylabel(r'$\mu$')
            leg_loc = 'lower right'
            if aspect == 0:
                aspect = 1/(2*pi)
            if aspect is not None:
                ax.set_aspect(aspect)
                if aspect < 1 and cbo is None:
                    cbo = 'horizontal'
                    leg_loc = 'upper right'

        if mode in ('shade', 'sphere'):
            if cbo is None:
                cbo = 'vertical'
            cb = fig.colorbar(cs, ax=ax, orientation=cbo)
            cb.set_label(sleg)

        x,y = np.array([[jpol[2], djpol[2], dwpol[2],jpolr[2], djpolr[2], dwpolr[2]],
                        [jpol[1], djpol[1], dwpol[1],jpolr[1], djpolr[1], dwpolr[1]]])
        if mode == 'sphere':
            x, y = m(x * r2d, 90 - y * r2d)
            x[x > 1e20] = np.nan
            y[y > 1e20] = np.nan
        elif mode in ('project', 'min', ):
            x[:] = vmin
            y = cos(y)
        else:
            x /= 2*pi
            y = cos(y)

        if fsh != 0:
            ax.plot(x[0:1], y[0:1], ls='none', marker='o', markersize=8, color="#00af7f",
                    clip_on=False,
                    label = r'$\vec{j}\,/\,\left|\vec{j}\right|$')
            ax.plot(x[1:2], y[1:2], ls='none', marker='^', markersize=8, color="#ffaf00",
                    clip_on=False,
                    label = r'$\frac{\mathrm{d}}{\mathrm{d}r}\vec{j}\,/\,\left|\frac{\mathrm{d}}{\mathrm{d}r}\vec{j}\right|$')
            if mode in ('sphere',):
                ax.plot(x[3:4], y[3:4], ls='none', marker='o', markersize=8, color="#00af7f",
                        clip_on=False, fillstyle = 'none')
                ax.plot(x[4:5], y[4:5], ls='none', marker='^', markersize=8, color="#ffaf00",
                        clip_on=False, fillstyle = 'none')
        else:
            ax.plot(x[0:1], y[0:1], ls='none', marker='o', markersize=8, color="#00af7f",
                    clip_on=False,
                    label = r'$\hat{\omega}$')
            #       label = r'$\vec{\omega}\,/\,\left|\vec{\omega}\right|$')
            if mode in ('sphere',):
                ax.plot(x[3:4], y[3:4], ls='none', marker='o', markersize=8, color="#00af7f",
                        clip_on=False, fillstyle = 'none')
        if ric != 0:
            ax.plot(x[2:3], y[2:3], ls='none', marker='*', markersize=10, color="#00afff",
                    clip_on=False,
                    label = r'$\frac{\mathrm{d}}{\mathrm{d}r}\vec{\omega}\,/\,\left|\frac{\mathrm{d}}{\mathrm{d}r}\vec{\omega}\right|$')
            if mode in ('sphere',):
                ax.plot(x[5:6], y[5:6], ls='none', marker='*', markersize=10, color="#00afff",
                        clip_on=False, fillstyle = 'none')
        leg = fig.legend(loc=leg_loc, framealpha = 1)

        # write parameters
        p1 = []
        if n2 != 0:
            p1 += [fr'$N^2={n2:5.2f}$']
        p1 += [fr'$r={ra:5.2f}$']
        if ric != 0:
            p1 += [fr'$Ri_\mathrm{{c}}={ric:5.2f}$']
        pv = []
        pd = []
        if fsh != 0:
            pv += [r'$\vec{{j}}=[{:5.2f},{:5.2f},{:5.2f}]$'.format(*j)]
            pd += [r'$\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\vec{{j}}=[{:5.2f},{:5.2f},{:5.2f}]$'.format(*dj)]
        if ric != 0:
            pv += [r'$\vec{{\omega}}=[{:5.2f},{:5.2f},{:5.2f}]$'.format(*w)]
            pd += [r'$\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\vec{{\omega}}=[{:5.2f},{:5.2f},{:5.2f}]$'.format(*dw)]
        sep = fr'$,\quad$'
        spar = sep.join(p1)
        if len(pv) > 0:
            sv = sep.join(pv)
            sp = sep.join(pd)
            spar = sep.join([spar,sv,sp])
        fig.text(0.005, 0.005, spar, color='gray', fontsize=6)


        if mode in ('project', 'min'):
            l = ax.axvline(0, c='r', ls=':',
                       label = f'stability\nlimit')
            ii = np.where(n2xa > 0)[0]
            x = n2xa
            y = mu[:,0]
            y[ii] = np.nan
            ax.fill_betweenx(y,x,0, color='#ffbbbb')
            ax.legend(handles=[l], loc='lower right', framealpha = 1)

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

# class ULogScale(ScaleBase):
#     name = 'ulog'
#     def get_transform(self):
#         pass
#     def set_default_locators_and_formatters(self):
#         pass
#     def limit_range_for_scale(self):
#         pass
#register_scale(ULogScale)

class MaxPlot(object):
    def __init__(
            self,
            n = 2**8,
            nt = 2**8,
            mode = 'angle', # angle | value | average | 2D
            scan = 'j', # j | w
            n2 = 1,
            ra = 1,
            fsh = 1,
            ric = 1/4,
            dja = 1,
            dwa = 1,
            ja = 1,
            wa = 1,
            framealpha = 1,
            calc_mode = 'min', # min | max
            ka = 2 / 3,
            ):

        j = np.asarray([0,0,1])*ja
        w = np.asarray([0,0,1])*wa

        t = theta = pi * np.arange(nt)/(nt-1)
        c = cos(t)
        s = sin(t)
        m = []
        a = []
        at = []
        mt = []
        mp = []
        tx = []
        mx = []
        muw = []
        tm = []
        tp = []

        n2sm=[]

        for cx,sx in zip(c,s):
            kwargs = dict()
            if scan == 'j':
                kwargs['dj'] = np.asarray([sx, 0, cx])*dja*ka
                kwargs['j'] = j*ka
            else:
                kwargs['dw'] = np.asarray([sx, 0, cx])*dwa
                kwargs['w'] = w
            s = DSIPlot(
                mode = 'calc',
                calc_mode = calc_mode,
                n    = n,
                n2   = n2,
                ric  = ric,
                ra   = ra,
                fsh  = fsh,
                **kwargs,
                )
            m.append(s.n2xm)
            a.append(s.n2xam)
            at.append(s.n2xamt)
            mt.append(s.n2xmt)
            mp.append(s.n2xmp)

            tx.append(s.t)
            mx.append(s.n2xtm)

            muw.append(s.muw)
            tm.append(s.tmin)
            tp.append(s.tmax)
            n2sm.append(s.n2m + s.n2wm + s.n2rm)

        mt = np.asarray(mt)
        mp = np.asarray(mp)
        at = np.asarray(at)
        a  = np.asarray(a)
        m  = np.asarray(m)
        muw= np.asarray(muw)
        tm = np.asarray(tm)
        tp = np.asarray(tp)
        n2sm = np.asarray(n2sm)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        sep = fr'$,\quad$'
        smuj=rf'$\mu=\cos\left(\vec{{j}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right)$'
        smuw=rf'$\mu=\cos\left(\vec{{\omega}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{\omega}}\right)$'
        saj=rf'$\sphericalangle\left(\vec{{j}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right)$'
        saw=rf'$\sphericalangle\left(\vec{{\omega}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{\omega}}\right)$'
        if scan == 'j':
            smu = smuj
            sa = saj
        else:
            smu = smuw
            sa = saw
        sc = calc_mode
        if sc == 'min':
            scl = 'minimum'
            scp = 'minima'
        else:
            scl = 'maximum'
            scp = 'maxima'
        spm=rf'$\phi_{{\{sc}}}$'
        snorm0 = r'\left|\vec{{j}}\right|\cdot\left|\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right|'
        snorm = rf'\,\left/\,\left({snorm0}\right)\right.'

        sdj = ''
        ni = 0
        if n2 != 0:
            sdj += rf'N^2'
            ni += 1
        if fsh != 0:
            if sdj != '':
                sdj += '+'
            sdj += rf'N^2_{{\Omega}}'
            ni += 1
        if ric != 0:
            sdj += rf'-Ri_\mathrm{{c}}\,\left(\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\vec{{\omega}}\times\vec{{r}}\right)^2'
            ni += 1
        sdjn = rf'{sdj}'
        sdjm = rf'$\{sc}\left(\,{sdjn}\,\right)$'
        sdjma = rf'location of $\{sc}\left(\,{sdjn}\,\right)$'
        st = rf'$\theta=\sphericalangle\left(\vec{{j}},\vec{{r}}\right)$'

        if mode == 'angle':
            ax.plot(t * r2d, mt * r2d, color = 'k', ls = '-', label = r'$\theta$')
            ax.plot(t * r2d, mp * r2d, color = 'k', ls = ':', label = r'$\phi$')
            ax.set_ylabel(sdjma)
            ax.set_xlabel(sa)
        elif mode == 'value':
            ax.plot(c, m, color = 'k', label = f'absolute {scl}')
            ax.plot(c, a, color = 'k', ls = ':', label = f'{sc}. of lat. average')
            if ni > 1:
                ax.plot(c, n2sm, color = 'k', ls = '--', label = f'sum of {scp}')
            ax.set_ylabel(sdjm)
            ax.set_xlabel(smu)
        elif mode == 'average':
            ax.plot(t * r2d, at * r2d, color = 'k', ls = ':',  label = r'$\theta$')
        elif mode == '2D':
            mx = np.asarray(mx).transpose()
            cmx = cos(mx)
            tx = np.asarray(tx[0])
            ctx = cos(tx)
            mag = np.max(np.abs(mx))
            vmin = -mag
            vmax = +mag
            cs = ax.pcolormesh(t*r2d, tx*r2d, mx, vmin=vmin, vmax=vmax, cmap = 'bwr_r')
            cb = fig.colorbar(cs, ax=ax)
            ax.set_xlabel(sa)
            ax.set_ylabel(st + sep + saw + sep + spm)
            cb.set_label(rf'{scl} of ${sdjn}$ on parallels at $\theta$')
            if ric != 0 and fsh == 0:
                # ax.plot(t * r2d, mt * r2d, color = 'k', ls = 'none', marker='.', label = 'absolute\nminimum')
                # ax.plot(t * r2d, (pi-mt) * r2d, color = 'k', ls = 'none', marker='.')

                # ax.plot(t * r2d, (tm) * r2d, color = 'c', ls = 'none', marker='+')
                # ax.plot(t * r2d, (tp) * r2d, color = 'c', ls = 'none', marker='x')

                ax.fill_between(t * r2d, tm * r2d, tp * r2d, hatch='x',
                                fc='none', ec='#7f7f7f7f',
                                label = f'absolute\n{scl}')
            else:
                ax.plot(t * r2d, mt * r2d, color = 'k', ls = '-', label = f'absolute\n{scl}')
                ax.plot(t * r2d, (pi-mt) * r2d, color = 'k', ls = '-')
            ax.plot(t * r2d, np.arccos(muw) * r2d, color = 'k', ls = ':', label = saw)
            phi = np.minimum(mp, 2*pi-mp)
            phi = np.minimum(phi, pi-phi)
            ax.plot(t[1:-1] * r2d, phi[1:-1] * r2d,
                    color = '#ffcf00', ls = 'none', ms = 8, marker='.', label = spm)
            framealpha = 1
        else:
            raise AttributeError(f'Unknown mode {mode}')

        # show parameters
        p1 = []
        if n2 != 0:
            p1 += [fr'$N^2={n2:5.2f}$']
        p1 += [fr'$r={ra:5.2f}$']
        if ric != 0:
            p1 += [fr'$Ri_\mathrm{{c}}={ric:5.2f}$']
        if scan == 'j':
            p1 += [fr'$j={ja:5.2f}$']
            p1 += [fr'$\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}j={dja:5.2f}$']
        else:
            p1 += [fr'$\omega={wa:5.2f}$']
            p1 += [fr'$\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\omega={dwa:5.2f}$']
        spar = sep.join(p1)
        fig.text(0.005, 0.005, spar, color='gray', fontsize=6)

        leg = ax.legend(loc='best', framealpha = framealpha)
        fig.tight_layout()
        self.fig = fig
        self.ax = ax

class EvolutionPlot(object):
    def __init__(
            self, *,
            align = True,
            j = [0,0,3.16],
            dj = np.array([.8, 0, .6])*2.5,
            #            dj = [0, 0, 1],
            w = None,
            dw = None,
            n = 2**8,
            fig = None,
            ax = None,
            alignphi = 0,
            proj = 'ortho',
            n2 = 10,
            k = 2/3, # gyration constant j = w * k * r **2
            ra = 1, # radius
            fsh = 1, # enable N2W term; set to 0 for 'classical' DSI
            ric = 1/4,
            vmin = None,
            vmax = None,
            mag = None,
            zinit = 1,
            nstep = 1,
            pause = None,
            ):

        # but we need t set up proper grid for integration
        #
        #
        #    X-------X-------X
        #    |       |       |
        #    |       |       |
        #    D   Z---D-->Z   D
        #    |       |       |
        #    |       |       |
        #    X-------X-------X
        #
        #   X: Mesh point for drawing
        #   Z: Field variables to integrate
        #   D: Staggered derivatves
        #      location of derivatives to use)
        #   ---D-->  one time step
        #
        # locations:
        #   f: grid locations (corners of the mesh)
        #   e: edge locations (derivatives) [omotted]
        #   c: centre locations (cell centre locations for values)

        thetaf, phif = mgrid[0:pi:(n+1)*1j, 0:2*pi:(n*2+1)*1j]

        self.subcycle = False
        theta = f2cx(thetaf[:,1:])
        phi   = f2cx(phif[:,1:])

        # [BEGIN NX] ---- this part here should is largely a copy ...
        r_hat = hat(theta, phi)
        r = ra * r_hat

        i = k * ra ** 2

        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            j = w * i
        else:
            j = np.asarray(j, dtype=np.float64)
            w = j / i
        if dw is not None:
            dw = np.asarray(dw, dtype=np.float64)
            dj = dw * i + 2 * j / ra
        else:
            dj = np.asarray(dj, dtype=np.float64)
            dw = dj / i - 2 * w / ra

        muj = np.dot( j, dj) / (norm(j)  * norm(dj))
        muw = np.dot( w, dw) / (norm(w)  * norm(dw))
        mud = np.dot(dw, dj) / (norm(dj) * norm(dw))

        # align such that j || z and dj in x-z plane

        assert align
        if align:
            j = np.array([0, 0, 1.]) * norm(j)
            w = np.array([0, 0, 1.]) * norm(w)
            dj = np.array([sqrt(1-muj**2)*cos(alignphi),
                           sqrt(1-muj**2)*sin(alignphi),
                           muj]) * norm(dj)
            dw = dj / i - 2 * w / ra

        # vector
        jdotrhat = np.tensordot( j, r_hat, (0,0))
        djdotrhat = np.tensordot(dj, r_hat, (0,0))
        jdotdj = np.dot(j, dj)

        ### not needed here
        # mu = cos(theta)
        # p = phi/(2 * pi)

        nw1 = jdotrhat * djdotrhat
        nw2 = jdotdj
        n2w = 2 * (nw2 - nw1) / (ra**3 * k**2) * fsh

        # DSI
        dwdlr = np.cross(dw, r, axis=0)
        dwdlr2 = np.sum(dwdlr * dwdlr, axis=0)

        # Richardson Criterion
        n2r = - ric * dwdlr2

        # generalised criterion
        n2x = n2 + n2w + n2r

        # diagnostic output
        # n2xp = np.max(n2x)
        # n2xm = np.min(n2x)
        # [END NX] ---- this part here should is largely a copy ...

        self.n2x = n2x

        # time evolution init

        self.dphi = phi[0,1]-phi[0,0]

        self.dt = self.dphi / norm(w)
        self.dz = np.exp(-1j * np.sqrt(np.array(n2x, dtype=np.complex)) * self.dt)
        self.z = np.full(n2x.shape, zinit, dtype=np.complex)


        # self.dz[:] = 1
        # self.z[-30:-20, 20:30] *= 2

        # plotting

        self.fig = fig

        self.i = 0
        self.n = n

        self.phif = phif
        self.thetaf = thetaf
        self.phi = phi
        self.theta = theta

        self.mag = mag
        self.vmin = vmin
        self.vmax = vmax

        self.pause = pause
        self.nstep = nstep

        self.proj = proj


    def draw(self, fig = None, pause = None):

        if fig is None:
             fig = self.fig
        if fig is None:
            fig = plt.figure()


        fig.clear()
        ax = fig.add_axes([0.05, 0.03, 0.7, 0.9])

        m = Basemap(
            projection=self.proj,
            ax=ax,
            lat_0=-37.907803,
            lon_0=145.133957 - np.mod(self.i, 2 * self.n) * self.dphi * r2d,
            )
        m.drawmeridians(np.arange(  0, 360, 30)) # grid every 30 deg
        m.drawparallels(np.arange(-90,  90, 30))
        m.drawcoastlines(color = '#ffffff3f')

        lon = r2d * self.phif
        lon[lon > 180] -= 360
        lat = 90 - r2d * self.thetaf

        cmap = 'PiYG'

        data = self.z.real
        data = np.roll(data, shift = -self.i, axis=1)

        if self.mag is None:
            mag = np.max(abs(self.z))
        else:
            mag = self.mag
        if self.vmin is None:
            vmin = -mag
        else:
            vmin = self.vmin
        if self.vmax is None:
            vmax = mag
        else:
            vmax = self.vmax

        # fix for pcolormesh (created pull request)
        data = pcm_fix(*m(lon,lat),data.copy())

        if False:
            cs = m.pcolor(lon, lat, data, vmin=vmax, vmax=vmin, cmap=cmap, latlon=True)
        else:
            cs = m.pcolormesh(lon, lat, data, vmin=vmin, vmax=vmax, cmap=cmap, latlon=True)

        leg_loc = 'upper right'
        stit = f'time = {time2human(self.i * self.dt)}'
        self.txt = fig.text(0.01, 0.99, stit, va='top', fontsize='large', color='k')

        cax = fig.add_axes([0.80,.05, 0.03, 0.9])
        sleg = r'Disturbance'
        cbo = 'vertical'

        self.cb = fig.colorbar(cs, cax=cax, orientation=cbo)
        self.cb.set_label(sleg)

        # n2x as conctours
        lon = r2d * self.phi
        lon[lon > 180] -= 360
        lat = 90 - r2d * self.theta
        data = self.n2x
        data = np.roll(data, shift = -self.i, axis=1)
        # we have to extend by one point
        lon = np.append(lon, lon[:,0:1], axis=1)
        lat = np.append(lat, lat[:,0:1], axis=1)
        data = np.append(data, data[:,0:1], axis=1)
        vmag = np.max(np.abs(data))
        vmin = -vmag
        vmax = +vmag
        cmap = 'coolwarm_r'
        ct = m.contour(
            lon, lat, data, latlon=True,
            cmap=cmap, vmin=vmin, vmax=vmax,
            )

        # done
        if pause is None:
            pause = self.pause
        if pause is not None:
            plt.pause(pause)
        self.ax = ax
        self.fig = fig

    def advance_timestep(
            self,
            nstep = None,
            target = None,
            ):
        # subcycle will only make sense of we interpolate dz, e.g., if
        # we were to compute on z locations and then do linear
        # interpolation
        if nstep is None:
            nstep = self.nstep
        if nstep is None:
            nstep = 1
        if target is not None:
            nstep = target - self.i
        print(f'[Integrator] Advancing by {nstep} steps.')
        for _ in range(nstep):
            self.z *= self.dz
            self.z = np.roll(self.z, shift=1, axis=1)
        self.i += nstep

    def draw_step(self, fig = None, target = None):
        self.advance_timestep(target = target)
        self.draw(fig = fig)

    @classmethod
    def movie(
            cls,
            n = 2**10,
            nframes = 1800,
            framerate = 60,
            size = (800, 600),
            dpi = 100,
            filename = '~/Downloads/growth.webm',
            parallel = True,
            fig = None,
            nstep = 2**2,
            **kwargs,
            ):
        print('Starting.')
        mfig = MPLBase(
            size = size,
            dpi = dpi,
            fig = fig,
            )
        fig = mfig.fig
        writer = MovieWriter(
            filename,
            delay = 1/framerate,
            getter = mfig.get_frame,
            parallel = parallel,
            )
        evo = cls(
            n = n,
            fig = fig,
            nstep = nstep,
            **kwargs,
            )
        for i in writer.range(nframes):
            print(f'Making frame {i+1} of {nframes}.')
            evo.advance_timestep()
            evo.draw()
        print('Done.')

    @classmethod
    def movie2(
            cls,
            n = 2**10,
            nframes = 3600,
            framerate = 60,
            size = (800, 600),
            dpi = 100,
            filename = '~/Downloads/growth2.webm',
            nstep = 2**2,
            **kwargs,
            ):
        ParallelMovie.make(
            filename,
            base = cls,
            bkwargs = dict(n = n, **kwargs),
            func = 'draw_step',
            data = 'target',
            values = np.arange(nframes) * nstep,
            canvas = MPLBase,
            ckwargs = dict(size=size, dpi=dpi),
            framerate = framerate,
            )


class MeridianPlot(object):
    def __init__(
            self,
            mus = 1,
            n = 2**10,
            fig = None,
            mode = 'angle',
            ):
        assert mode in ('angle', 'cos',)
        theta, phi = mgrid[0:2*pi:n*2j, 0:0:1j]
        r_hat = hat(theta,phi)

        if fig is None:
            fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)

        w_hat = np.array([1, 0, 0])
        dw_hat = np.array([mus, 0, np.sqrt(1-mus**2)])

        muj = np.tensordot( w_hat, r_hat, (0,0))
        muw = np.tensordot( dw_hat, r_hat, (0,0))

        if mode == 'angle':
            x = arccos(muj) * r2d
            y = arccos(muw) * r2d
        else:
            x = muj
            y = muw
        ax.plot(x.flat, y.flat, linestyle='-')
        ax.set_aspect(1)
        if mode == 'angle':
            sym = r'\theta'
            ts = arccos(mus) * r2d
            tit = fr'$\theta^*={int(ts):+3d}$'
        else:
            sym = r'\mu'
            tit = fr'$\mu^*={mus:+4.2f}$'
        ax.set_xlabel(fr'${sym}_{{\mathrm{{j}}}}$')
        ax.set_ylabel(fr'${sym}_{{\mathrm{{w}}}}$')
        ts = arccos(mus) * r2d
        fig.text(0.02,0.02, tit,
                 transform=fig.transFigure)
        fig.text(0.98,0.02, 'Points on meridian',
                 ha='right',
                 transform=fig.transFigure)
        # fig.tight_layout()
        ax.set_position([0.14, 0.14, 0.85, 0.85])

    @classmethod
    def movie(cls, ns = 200):
        mus = mgrid[-1:1:ns*1j]
        mfig = MPLBase(size=(600,600), dpi=100)
        writer = MovieWriter(
            '~/Downloads/mert.webm',
            getter = mfig.get_frame,
            parallel = True,
            )
        for x in writer.iter(mus):
            mfig.fig.clear()
            cls(mus = x, fig = mfig.fig)

class AnglePlot(object):
    def __init__(
            self,
            mus = 1,
            n = 2**8,
            fig = None,
            ):
        theta, phi = mgrid[0:pi:n*1j, 0:2*pi:n*2j]
        r_hat = hat(theta,phi)

        if fig is None:
            fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)

        w_hat = np.array([1, 0, 0])
        dw_hat = np.array([mus, 0, np.sqrt(1-mus**2)])

        muj = np.tensordot( w_hat, r_hat, (0,0))
        muw = np.tensordot( dw_hat, r_hat, (0,0))

        ax.plot(muj.flat, muw.flat, linestyle='none', marker='.')
        ax.set_aspect(1)
        ax.set_xlabel(r'$\mu_{\mathrm{j}}$')
        ax.set_ylabel(r'$\mu_{\mathrm{w}}$')
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        fig.text(0.02,0.02, fr'$\mu^*={mus:+4.2f}$',
                 transform=fig.transFigure)
        ax.set_position([0.14, 0.14, 0.85, 0.85])
        #fig.tight_layout()

    @classmethod
    def movie(cls, ns = 200):
        # ParallelMovie.make(
        PoolMovie.make(
            '~/Downloads/mus.webm',
            func = cls,
            values = mgrid[-1:1:ns*1j],
            canvas = MPLBase,
            ckwargs = dict(size=(600,600), dpi=100),
            )

class Movie(object):
    def __init__(
            self,
            nframes = 360,
            framerate = 60,
            size = (800, 600),
            filename = '~/Downloads/DSI.webm',
            parallel = False,
            dpi = 100,
            fig = None,
            **kwargs,
             ):

        print('Starting.')

        theta = pi * np.arange(nframes)/(nframes-1)

        mfig = MPLBase(
            size = size,
            dpi = dpi,
            fig = fig,
            )
        fig = mfig.fig

        writer = MovieWriter(
            filename,
            delay = 1/framerate,
            getter = mfig.get_frame,
            parallel = parallel,
            )
        j = [0,0,-1]

        for i,t in writer.enumerate(theta):
            print(f'Making frame {i+1} of {len(theta)}.')
            dj = -np.array([sin(t),0,cos(t)])
            fig.clear()
            DSIPlot(dj=dj, j=j, fig = fig, **kwargs)

        print('Done.')


class FrameGenerator(ParallelMovieFrames):
    def __init__(self, size, dpi, canvas = None, **kwargs):
        self.mfig = MPLBase(
            size = size,
            dpi = dpi,
            canvas = canvas,
            )
        self.fig = self.mfig.fig
        self.kwargs = kwargs

    def draw(self, i, t):
        print(f'Making frame {i+1}.')
        dj = np.array([sin(t),0,cos(t)])
        j = [0,0,1]
        self.fig.clear()
        DSIPlot(dj=dj, j=j, fig = self.fig, **self.kwargs)
        return self.mfig.get_frame()

    def close(self):
        self.mfig.close()


class MovieParallel(object):
    def __init__(
            self,
            nframes = 720,
            framerate = 60,
            size = (800, 600),
            filename = '~/Downloads/DSI.webm',
            nparallel = None,
            dpi = 100,
            pool = True,
            **kwargs,
             ):

        # this is if we do a loop - exclude otherwise duplicate frame
        theta = 2 * pi * np.arange(nframes)/nframes

        print('Starting.')

        if pool:
            P = PoolMovie
        else:
            P = ParallelMovie

        p = P(
            filename,
            nparallel = nparallel,
            delay = 1/framerate,
            generator = FrameGenerator,
            gargs = (size, dpi),
            gkwargs = kwargs,
            )
        p.run(theta)

        print('Done.')


def fw(pw, mwdw, mw, mdw, *, ric=1/4, fsh = 1):
    return (- ric * pw**2 * (1 - mdw**2)
            + fsh * (2 * pw * (mwdw - mw * mdw) + 4 * (1 - mw**2)))

def fj(pj, mjdj, mj, mdj, *, ric=1/4, fsh = 1):
    return (- ric * (pj**2 * (1 - mdj**2) + 4 * (1 - mj**2))
            + (2 * fsh + 4 * ric) * pj * (mjdj - mj * mdj))

def theta_min_fw(pw, twdw, *, ric=1/4, fsh = 1):
    phi = 0.5 * arctan2(
        - pw**2 * sin(2 * twdw) * ric + 2 * pw * sin(twdw) * fsh,
        - pw**2 * cos(2 * twdw) * ric + (2 * pw * cos(twdw) + 4) * fsh )
    phi = np.mod(phi, pi)
    return phi

def theta_min_fw_c(pw, twdw, *, ric=1/4, fsh = 1):
    A = 0.25 * (ric * pw**2 * np.exp(-2j * twdw) - fsh * (2 * pw * np.exp(-1j * twdw) + 4))
    phiA = np.angle(A) # same as np.log(A).imag but does not blow up for real(A) == 0
    phi = (pi - phiA) * 0.5 # track the minimum branch
    phi = np.mod(phi, pi) # this is not needed for solution, just normalises
    return phi

def tw_min(pw, mwdw, *, ric=1/4, fsh = 1):
    twdw = arccos(mwdw)
    tw = theta_min_fw(pw, twdw, ric = ric, fsh = fsh)
    return tw

def fw_min(pw, mwdw, *, ric=1/4, fsh = 1):
    twdw = arccos(mwdw)
    tw = theta_min_fw(pw, twdw, ric = ric, fsh = fsh)
    mw = cos(tw)
    mdw = cos(tw - twdw)
    return fw(pw, mwdw, mw, mdw, ric = ric, fsh = fsh)

def fw_mint(pw, twdw, *, ric=1/4, fsh = 1):
    tw = theta_min_fw(pw, twdw, ric = ric, fsh = fsh)
    mw = cos(tw)
    mdw = cos(tw - twdw)
    mwdw = cos(twdw)
    return fw(pw, mwdw, mw, mdw, ric = ric, fsh = fsh)

def theta_min_fj(pj, tjdj, *, ric=1/4, fsh = 1):
    phi = 0.5 * arctan2(
        -ric * pj**2 * sin(2 * tjdj) + pj * (2 * fsh + 4 * ric) * sin(tjdj),
        -ric * pj**2 * cos(2 * tjdj) + pj * (2 * fsh + 4 * ric) * cos(tjdj) - 4 * ric)
    phi = np.mod(phi, pi)
    return phi

def theta_min_fj_c(pj, tjdj, *, ric=1/4, fsh = 1):
    A = 0.25 * ric * pj**2 * np.exp(-2j * tjdj) - 0.25 * pj * (2 * fsh + 4 * ric) * np.exp(-1j * tjdj) + ric
    phiA = np.angle(A) # same as np.log(A).imag but does not blow up for real(A) == 0
    phi = (pi - phiA) * 0.5 # track the minimum branch
    phi = np.mod(phi, pi) # this is not needed for solution, just normalises
    return phi

def tj_min(pj, mjdj, *, ric=1/4, fsh = 1):
    tjdj = arccos(mjdj)
    tj = theta_min_fj(pj, tjdj, ric = ric, fsh = fsh)
    return tj

def fj_mint(pj, tjdj, *, ric=1/4, fsh = 1):
    tj = theta_min_fj(pj, tjdj, ric = ric, fsh = fsh)
    mj = cos(tj)
    mdj = cos(tj - tjdj)
    mjdj = cos(tjdj)
    return fj(pj, mjdj, mj, mdj, ric = ric, fsh = fsh)

def fj_min(pj, mjdj, *, ric=1/4, fsh = 1):
    tjdj = arccos(mjdj)
    tj = theta_min_fj(pj, tjdj, ric = ric, fsh = fsh)
    mj = cos(tj)
    mdj = cos(tj - tjdj)
    return fj(pj, mjdj, mj, mdj, ric = ric, fsh = fsh)

def f2c(f):
    """interpolate from corners to centre"""
    return 0.25 * (f[:-1,:-1] + f[1:,:-1] + f[:-1,1:] + f[1:,1:])

def f2cx(f):
    """interpolate from corners to edges in on axis 0"""
    return 0.5 * (f[:-1,:] + f[1:,:])

def pcm_fix(x,y,z, thres=1.e20):
    nx,ny = x.shape
    ii = (
        (x[:-1,:-1] > 1e20) |
        (x[1:,:-1] > 1e20) |
        (x[:-1,1:] > 1e20) |
        (x[1:,1:] > 1e20) |
        (y[:-1,:-1] > 1e20) |
        (y[1:,:-1] > 1e20) |
        (y[:-1,1:] > 1e20) |
        (y[1:,1:] > 1e20)
        )
    z[:nx-1,:ny-1][ii] = np.nan
    return z

def test_solution(pw = 1, pj = None, n = 2**6):
    # complex max test
    M = MaxPlot
    if pj is None:
        m = M(mode='angle', dwa=pw, scan='w', n = n)
        c = theta_min_fw_c
        t = theta_min_fw
        p = pw
    else:
        m = M(mode='angle', dja=pj, scan='j', n = n)
        c = theta_min_fj_c
        t = theta_min_fj
        p = pj

    x = np.mgrid[0:pi:1000j]
    ax = m.ax
    leg0 = ax.legend_
    l=[]
    l.extend(ax.plot(x*180/np.pi,np.real(c(p,x))*180/np.pi, color='g', label=r'$z$'))
    l.extend(ax.plot(x*180/np.pi,t(p,x)*180/np.pi, ls=':', lw=5, color='r', label=r'$\varphi$'))
    ax.legend(handles=l, loc='upper left', framealpha = 1)
    ax.add_artist(leg0)

class MinPlot(object):
    def __init__(
            self,
            n2 = 0,
            w = 1,
            j = 1,
            xmag = 3,
            xmin = None, # -2.5
            xmax = None, # +3.0
            n = 2**10,
            ric = 1/4,
            fsh = 1,
            mode = None, # theta | None = f
            ymode = 'theta', # theta | None
            xmode = None,  # tan | erf | None
            xrange = None,
            mrange = None,
            nscale = 0, # 0, <0, >0, None
            rmode = 'j', # j | w
            figsize = None,
            dpi = None,
            ):

        assert n2 == 0 and w == 1 and j == 1, 'need to replace f/g in color bar legend'

        if mode is None:
            mode = 'f'

        if mrange is not None:
            p0, p1 = arccos(mrange)
        else:
            p0, p1 = 0, pi

        if xrange is not None:
            x0, x1 = xrange
        else:
            x0, x1 = -xmag, xmag
        if xmin is not None:
            x0 = xmin
        if xmax is not None:
            x1 = xmax

        pl, tx = np.mgrid[x0:x1:1j*(n*2+1), p0:p1:1j*(n+1)]
        p = 10**pl

        mx = cos(tx)

        mxc = f2c(mx)
        pc = f2c(p)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1,1,1)

        vmin = None
        vmax = None

        xr = None
        yr = None

        if ric == 0:
            smode = r'{{\Omega}}'
            sfmode = fr'_{smode}'
            stmode = fr'{smode},'
        elif fsh == 0:
            smode = r'{{\mathrm{{R}}}}'
            sfmode = fr'_{smode}'
            stmode = fr'{smode},'
        else:
            smode = ''
            stmode = ''
            sfmode = ''

        if rmode == 'w':
            if mode == 'theta':
                minval = tw_min(pc, mxc, ric = ric, fsh = fsh)
            else:
                minval = fw_min(pc, mxc, ric = ric, fsh = fsh)
            sfun = f'f'
            spvar = fr'$\breve\theta$'
            sxvar = f't'
        else:
            spvar = fr'$\tilde\theta$'
            sxvar = f's'
            if mode == 'theta':
                minval = tj_min(pc, mxc, ric = ric, fsh = fsh)
            else:
                minval = fj_min(pc, mxc, ric = ric, fsh = fsh)
            sfun = f'g'
        # sfunp = sfun + rf'\left({sxvar}\right)'
        sfunp = sfun + fr'_{{{stmode}\min}}'

        if ymode == 'theta':
            ylab = spvar
            y = tx * 180/pi
            yr = np.array([p1, p0]) * 180 / pi
            ax.axhline(90, ls='--', color='k')
        else:
            ylab = fr'$\cos\,{spvar}$'
            y = mx
            yr = cos(np.array([p0,p1]))
            ax.axhline(0, ls='--', color='k')

        if xmode == 'tan':
            xlab = rf'$\arctan(\log\,{sxvar})$'
            x = arctan(pl)# * 2 / pi
        elif xmode == 'erf':
            xmag = np.max(np.abs(pl))
            scale = xmag / 2
            xlab = rf'$\mathrm{{erf}}(\log({sxvar})\,/\,{scale})$'
            x = erf(pl / scale)
        else:
            xlab = rf'$\log\,{sxvar}$'
            x = pl

        if mode == 'theta':
            vmin = 0
            vmax = 180
            z = np.mod(minval, pi) * 180 / pi
            cs = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap = 'twilight')
            sleg = fr'$\theta_{{{stmode}\min}}$'
        else:
            n2x = n2 + w**2*minval
            if nscale is None:
                z = arctan(n2x) * 2 / pi
                vmin = -1
                vmax = +1
                sleg = fr'$\frac{{2}}{{\pi}}\,\arctan({sfunp})$'
            elif nscale == 't':
                if rmode == 'w':
                    #z = n2x * (1 + pc**2*2**(-3)) / ((pc**2) * (1 + pc)))
                    #z = n2x * (1 + pc *2**(-3)) / pc**2
                    #z = n2x * (1 + 0*pc + pc**2*2**(-3)) / ((pc**2) * (1 + pc))
                    z = n2x / pc * 0.125
                    sleg = fr'${sfunp}\,\times\,\left(8\,{sxvar}\right)^{{-1}}$'
                    vmag = 0.5
                else:
                    z = n2x / pc * 0.125
                    sleg = fr'${sfunp}\,\times\,\left(8\,{sxvar}\right)^{{-1}}$'
                    vmag = 0.5
                vmin = -vmag
                vmax = 0
            elif nscale == 0:
                if rmode == 'w':
                    z = n2x / pc**2
                    sleg = fr'${sfunp}\,\times\,{sxvar}^{{-2}}$'
                    vmag = 0.5
                else:
                    z = n2x / (pc**2 + 12 * pc + 4)
                    sleg = fr'${sfunp}\,/\,\left({sxvar}^{{2}}+12\,{sxvar}+4\right)$'
                    vmag = 0.5
                vmin = -vmag
                vmax = 0
            elif nscale < 0:
                z = n2x * pc**nscale
                sleg = fr'${sfunp}\,\times\,{sxvar}^{{{nscale}}}$'
                vmag = np.max(np.abs(z))
                vmin = -vmag
                vmax = +vmag
            else:
                if nscale == 1:
                    sleg = fr'${sfunp}$'
                else:
                    sleg = fr'${sfunp}/{nscale})$'
                z = n2x / nscale
                vmag = np.max(np.abs(z))
                vmin = -vmag
                vmax = +vmag
            cf = ColorBlend(
                ('white', 'red'),
                func=lambda x: (x+1.e-99)**(1/1.5),
                )
            cf = ColorBlendWave(
                cf,
                'black',
                amplitude=0.02,
                power=0,
                nwaves=25,
                )

            cs = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap = cf)
            sleg += '\n' + r'$\longleftarrow$ more unstable'

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        ax.set_ylim(xr)
        ax.set_ylim(yr)

        cbo = 'vertical'
        cb = fig.colorbar(cs, ax=ax, orientation=cbo)
        cb.set_label(sleg)

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

    @classmethod
    def make_pub_plots(
            cls,
            figsize = None,
            dpi = None,
            base = os.path.expanduser('~/LaTeX/oblique'),
            ):
        x = cls(xmin=-2.5, xmax=3, rmode='w', mode='theta', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'theta-min-pw.png'))
        x.fig.canvas.manager.destroy()
        x = cls(xmin=-2.5, xmax=3, rmode='j', mode='theta', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'theta-min-pj.png'))
        x.fig.canvas.manager.destroy()
        x = cls(xmin=-2.5, xmax=3, rmode='w', mode=None, figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'fw-min-pw.png'))
        x.fig.canvas.manager.destroy()
        x = cls(xmin=-2.5, xmax=3, rmode='j', mode=None, figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'fj-min-pj.png'))
        x.fig.canvas.manager.destroy()

    @classmethod
    def make_pub_plots_extra(
            cls,
            figsize = None,
            dpi = None,
            base = os.path.expanduser('~/LaTeX/oblique'),
            ):
        x = cls(xmin=-2.5, xmax=3, ric=0, rmode='w', mode='theta', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'tSH-min-pw.png'))
        x.fig.canvas.manager.destroy()
        x = cls(xmin=-2.5, xmax=3, fsh=0, rmode='w', mode='theta', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'tDSI-min-pw.png'))
        x.fig.canvas.manager.destroy()
        x = cls(xmin=-2.5, xmax=3, ric=0, rmode='j', mode='theta', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'tSH-min-pj.png'))
        x.fig.canvas.manager.destroy()
        x = cls(xmin=-2.5, xmax=3, fsh=0, rmode='j', mode='theta', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'tDSI-min-pj.png'))
        x.fig.canvas.manager.destroy()
        x = cls(xmin=-2.5, xmax=3, ric=0, rmode='w', mode=None, figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'fSH-min-pw.png'))
        x.fig.canvas.manager.destroy()
        x = cls(xmin=-2.5, xmax=3, fsh=0, rmode='w', mode=None, figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'fDSI-min-pw.png'))
        x.fig.canvas.manager.destroy()


    @classmethod
    def make_pub_plots_extra2(
            cls,
            figsize = None,
            dpi = None,
            base = os.path.expanduser('~/LaTeX/oblique'),
            ):
        x = cls(xmin=-2.5, xmax=3, ric=0, rmode='w', mode='f', nscale='t', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'tSH-min-pw-t.png'))
        x.fig.canvas.manager.destroy()



class AlignedPlot(object):
    def __init__(
            self,
            rmode = 'w', # j | w
            xmag = 5.2,
            xmin = -5,
            xmax = None,
            xrange = None,
            n = 2**10,
            n2 = 0,
            w = 1,
            mode = None, # theta | None
            ymode = 't', # t | None
            xmode = None, # None | 'tan'
            verbose = False,
            numerical = False,
            aligned = False,
            ka = 2 / 3,
            showall = True,
            figsize = None,
            dpi = None,
            ):
        if xrange is not None:
            x0, x1 = xrange
        else:
            x0, x1 = -xmag, xmag

        if xmin is not None:
            x0 = xmin
        if xmax is not None:
            x1 = xmax

        mx = np.tile(1., n+1)
        pl = np.mgrid[x0:x1:1j*(n+1)]
        p = 10**pl

        p = np.array((p[::-1]).tolist() + p.tolist())
        mx = np.array((-mx[::-1]).tolist() + mx.tolist())

        if rmode == 'w':
            if mode == 'theta':
                minval = tw_min(pc, mxc, ric = ric, fsh = fsh)
                if showall:
                    minvalw = tw_min(p, mx, ric = 0)
                    minvalr = tw_min(p, mx, fsh = 0)
            else:
                minval = fw_min(pc, mxc, ric = ric, fsh = fsh)
                if showall:
                    minvalw = fw_min(p, mx, ric = 0)
                    minvalr = fw_min(p, mx, fsh = 0)
            sxvar = 't'
            sfun = 'f'
        else:
            if mode == 'theta':
                minval = tj_min(pc, mxc, ric = ric, fsh = fsh)
                if showall:
                    minvalw = tj_min(p, mx, ric = 0)
                    minvalr = tj_min(p, mx, fsh = 0)
            else:
                minval = fj_min(pc, mxc, ric = ric, fsh = fsh)
                if showall:
                    minvalw = fj_min(p, mx, ric = 0)
                    minvalr = fj_min(p, mx, fsh = 0)
            sxvar = 's'
            sfun = 'g'
        #sfunp = sfun + rf'\left({sxvar}\right)'
        sfunp = sfun + rf'_{{\min}}'
        sxcoord = fr'{sxvar}^*'

        # now with sign (!)
        ps = p * mx

        laba = 'analytical'
        labn = 'numerical'
        labl = 'aligend'
        if numerical:
            minnum = []
            for px in ps:
                if rmode == 'w':
                    kw = dict(
                        w = [0, 0, 1],
                        dw = [0, 0, px],
                        )
                else:
                    # j is only shell average w
                    kw = dict(
                        j = [0, 0, ka],
                        dj = [0, 0, px * ka],
                        )
                d = DSIPlot(
                    calc_mode = 'min',
                    mode = 'calc',
                    n = n,
                    n2 = n2,
                    align = True,
                    **kw,
                    )
                if mode == 'theta':
                    minnum.append(d.n2xmt)
                else:
                    minnum.append(d.n2xm)
                if verbose:
                    print(p, d.n2xm, d.n2xmt*180/pi)
        elif aligned:
            if mode == 'theta':
                minali = np.tile(pi/2, p.shape)
            else:
                if rmode == 'w':
                    minali  = fw(p, mx, mw = 0, mdw = 0)
                    # minaliw = fw(p, mx, mw = 0, mdw = 0, ric = 0)
                    # minalir = fw(p, mx, mw = 0, mdw = 0, fsh = 0)
                else:
                    minali  = fj(p, mx, mj = 0, mdj = 0)
                    # minaliw = fj(p, mx, mj = 0, mdj = 0, ric = 0)
                    # minalir = fj(p, mx, mj = 0, mdj = 0, fsh = 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1,1,1)

        ytrans = lambda y: y

        if mode == 'theta':
            ytrans = lambda y: y * 180/pi
            ylab = r'$\theta$ of minimum'
            ax.axhline(ytrans(pi/2), ls=':')
        else:
            if ymode == 't':
                if rmode == 'w':
                    ytrans = lambda y: y / p**2
                    ylab = fr'${sfunp}\,\times\,{sxvar}^{{-2}}$'
                else:
                    ytrans = lambda y: y / (p**2 + 4 + 12 * p)
                    ylab = fr'${sfunp}\,/\,\left({sxvar}^{{2}}+12\,{sxvar}+4\right)$'
                ax.axhline(0, ls='-', color='#DFDFDF', lw = 5)
            else:
                ylab = fr'${sfunp}$'
                loglim = 0.1
                ax.set_yscale('symlog',linthreshy = loglim)
                ax.axvspan(-loglim, loglim,color='#DFDFDF')

        y  = ytrans(minval)

        if xmode == 'tan':
            x = arctan(ps)
            xlab = rf'$\arctan({sxcoord})$'
        elif xmode == 'erf':
            x = erf(ps)
            xlab = rf'$\mathrm{erf}({sxcoord})$'
        else:
            x = ps
            loglim = 1
            ax.set_xscale('symlog',linthreshx = loglim)
            ax.axvspan(-loglim, loglim,color='#FFFFCF')
            #ax.set_xscale('symlog',linthreshx=1)
            xlab = rf'${sxcoord}$'

        if numerical:
            ax.plot(x, y, marker='.', ls = 'none', label=laba)
            y2 = ytrans(np.array(minnum))
            ax.plot(x, y2, marker='+', ls = 'none', label=labn)
            ax.legend(loc='best')
        elif aligned:
            ax.plot(x, y, ls = '-', label=laba)
            y3 = ytrans(minali)
            ax.plot(x, y3, ls = '--', label=labl)
            ax.legend(loc='best')
        else:
            ax.plot(x, y, ls = '-', label = f'${sfun}$')
            if showall:
                yr  = ytrans(minvalr)
                ax.plot(x, yr, ls = ':', color = 'red', label = f'${sfun}_{{\mathrm{{R}}}}$')
                yw  = ytrans(minvalw)
                ax.plot(x, yw, ls = ':', color = 'green', label = f'${sfun}_{{\Omega}}$')
                ax.legend(loc='best')
            ax.set_ylim(-0.525, +0.025)


        ax.set_xlim(min(x), max(x))

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        # replace by good settings instead
        fig.tight_layout()

        self.fig = fig
        self.ax = ax

    @classmethod
    def make_pub_plots(
            cls,
            figsize = None,
            dpi = None,
            figscale = None,
            base = os.path.expanduser('~/LaTeX/oblique'),
            ):
        if figsize is None:
            figsize = [6.4, 4.8]
        figsize = np.array(figsize)
        if figscale is not None:
            figsize *= figscale
        x = cls(xmag=5.2, rmode='w', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'theta-min-aligned-pw.pdf'))
        x.fig.canvas.manager.destroy()
        x = cls(xmag=5.2, rmode='j', figsize=figsize, dpi=dpi)
        x.fig.savefig(os.path.join(base, 'theta-min-aligned-pj.pdf'))
        x.fig.canvas.manager.destroy()


class MGPlot(object):
    def __init__(
            self,
            mode = 'w', # w | j
            xmag = 2.6,
            xmin = None,
            xmax = None,
            xrange = None,
            xmode = None,
            ymode = None,
            zmode = None,
            mrange = None,
            n = 2**10,
           ):
        if mrange is not None:
            p0, p1 = arccos(mrange)
        else:
            p0, p1 = 0, pi
        if xrange is not None:
            x0, x1 = xrange
        else:
            x0, x1 = -xmag, xmag
        if xmin is not None:
            x0 = xmin
        if xmax is not None:
            x1 = xmax

        pl, tx = np.mgrid[x0:x1:1j*(n*2+1), p0:p1:1j*(n+1)]
        p = 10**pl

        mx = cos(tx)

        mxc = f2c(mx)
        pc = f2c(p)
        txc = f2c(tx)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        vmin = None
        vmax = None

        sgvar = r'\mu_{{\mathrm{{g}}}}'
        shvar = r'\theta_{{\mathrm{{g}}}}'

        if mode == 'w':
            mg = (pc + 2 * mxc) / sqrt(pc**2 + 4 * pc * mxc + 4)
            my = (pc * mxc + 2) / sqrt(pc**2 + 4 * pc * mxc + 4)
            sxvar = 't'
            spvar = r'\breve\theta'
            smvar = r'\breve\mu'
            sqvar = r'\tilde\theta'
            snvar = r'\tilde\mu'
        else:
            mg = (pc - 2 * mxc) / sqrt(pc**2 - 4 * pc * mxc + 4)
            my = (pc * mxc - 2) / sqrt(pc**2 - 4 * pc * mxc + 4)
            sxvar = 's'
            spvar = r'\tilde\theta'
            smvar = r'\tilde\mu'
            sqvar = r'\breve\theta'
            snvar = r'\breve\mu'

        if xmode is None:
            xlab = rf'$\log\,{sxvar}$'
            x = np.log10(p)
        xr = None

        if ymode is None:
            y = tx*180/pi
            ylab = fr'${spvar}$'
            ax.axhline(90, ls='--', color='k')
            yr = np.array([p1, p0]) * 180 / pi
        elif ymode == 'mu':
            y = mx
            ylab = fr'${smvar}$'
            ax.axhline(0, ls='--', color='k')
            yr = cos(np.array([p0, p1]))

        if zmode is None:
            z = np.arccos(mg) * 180 / pi
            vmin = 0
            vmax = 180
            sleg = rf'${shvar}$'
            cm = 'bwr'
        elif zmode == 'mu':
            z = mg
            vmin = -1
            vmax = +1
            sleg = rf'${sgvar}$'
            cm = 'bwr_r'
        elif zmode == 'my':
            z = my
            vmin = -1
            vmax = +1
            sleg = rf'${snvar}$'
            cm = 'bwr_r'
        elif zmode == 'ty':
            z = np.arccos(my) * 180 / pi
            vmin = 0
            vmax = +180
            sleg = rf'${sqvar}$'
            cm = 'bwr_r'
        elif zmode == 'dmy':
            z = mxc - my
            vmin = -2
            vmax = +2
            sleg = rf'${smvar}-{snvar}$'
            cm = 'bwr_r'
        elif zmode == 'dty':
            ty = np.arccos(my)
            z = (txc - ty) * 180 / pi
            vmin = -180
            vmax = +180
            sleg = rf'${spvar}-{sqvar}$'
            cm = 'bwr_r'
        elif zmode == 'dt':
            z = (txc - np.arccos(mg)) * 180 / pi
            # z = np.mod(z + 180, 360) - 180
            # vmin = -180
            # vmax = +180
            sleg = rf'${spvar}-{shvar}$'
            cm = 'bwr_r'

        cs = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap = cm)
        cbo = 'vertical'
        cb = fig.colorbar(cs, ax=ax, orientation=cbo)
        cb.set_label(sleg)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        ax.set_ylim(xr)
        ax.set_ylim(yr)

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

class CorrPlot(object):
    def __init__(
            self,
            mode = 'w', # w | j
            pmag = 2.6,
            pmin = None,
            pmax = None,
            prange = None,
            xmode = None,
            ymode = None,
            zmode = 'p',
            mrange = None,
            n = 2**10,
            zbmode = None,
            fig = None,
            ax = None,
            draw_cb = True,
            ax_pos = None,
            cb_pos = None,
            tight = None,
            label = True,
            ):
        if mrange is not None:
            t0, t1 = arccos(mrange)
        else:
            t0, t1 = 0, pi
        if prange is not None:
            p0, p1 = xrange
        else:
            p0, p1 = -pmag, pmag
        if pmin is not None:
            p0 = pmin
        if pmax is not None:
            p1 = pmax

        if fig is None:
            figsize = np.array([6.4, 4.8])
            if zmode == 'p' and zbmode is not None:
                figsize += np.array([-1.095,+1.24])
            fig = plt.figure(figsize=figsize)
            if tight is None:
                tight = True
        if ax is None:
            if ax_pos is not None:
                ax = fig.add_axes(ax_pos)
            else:
                ax = fig.add_subplot(1,1,1)
                if tight is None:
                    tight = True

        pl, tx = np.mgrid[p0:p1:1j*(n*2+1), t0:t1:1j*(n+1)]
        p = 10**pl

        mx = cos(tx)

        mxc = f2c(mx)
        pc = f2c(p)
        txc = f2c(tx)

        vmin = None
        vmax = None

        if mode == 'w':
            # mg = (pc + 2 * mxc) / sqrt(pc**2 + 4 * pc * mxc + 4)
            # my = (pc * mxc + 2) / sqrt(pc**2 + 4 * pc * mxc + 4)
            spvar = 't'
            stvar = r'\breve\theta'
            smvar = r'\breve\mu'
            suvar = r'\tilde\theta'
            snvar = r'\tilde\mu'
            sqvar = 's'
            q = sqrt(p**2 + 4 * p * mx+4)
            n = (p * mx + 2) / q
        else:
            # mg = (pc - 2 * mxc) / sqrt(pc**2 - 4 * pc * mxc + 4)
            # my = (pc * mxc - 2) / sqrt(pc**2 - 4 * pc * mxc + 4)
            sqvar = 't'
            suvar = r'\breve\theta'
            snvar = r'\breve\mu'
            stvar = r'\tilde\theta'
            smvar = r'\tilde\mu'
            spvar = 's'
            q = sqrt(p**2 - 4 * p * mx+4)
            n = (p * mx - 2) / q

        if xmode is None:
            xlab = rf'$\log\,{spvar}$'
            x = np.log10(p)
        xr = None

        if ymode is None:
            y = tx*180/pi
            ylab = fr'${stvar}$'
            ax.axhline(90, ls='--', color='k')
            yr = np.array([t1, t0]) * 180 / pi
        elif ymode == 'mu':
            y = mx
            ylab = fr'${smvar}$'
            ax.axhline(0, ls='--', color='k')
            yr = cos(np.array([t0, t1]))

        if zmode == 'p':
            z = np.log10(q)
            sleg = rf'$\log\,{sqvar}$'
            cm = 'plasma'
            cm = ColorBlendWave(
                cm,
                'black',
                amplitude=0.02,
                power=0,
                nwaves=25,
                )
            svar = rf'${sqvar}$'
        elif zmode == 'm':
            z = n
            vmin = -1
            vmax = +1
            sleg = rf'${snvar}$'
            cm = 'PRGn'
            cm = ColorBlendWave(
                cm,
                'black',
                amplitude=0.02,
                power=0,
                nwaves=25,
                )
            svar = rf'${snvar}$'
        elif zmode == 't':
            z = arccos(n) * 180 / pi
            vmin = 0
            vmax = 180
            sleg = rf'${suvar}$'
            cm = 'PiYG_r'
            cm = ColorBlendWave(
                cm,
                'black',
                amplitude=0.02,
                power=0,
                nwaves=25,
                )
            svar = rf'${suvar}$'

        if zmode == 'p' and zbmode in ('prange', ):
            vmin = np.min(x)
            vmax = np.max(x)
        cs = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap = cm)
        if draw_cb:
            if zmode == 'p' and zbmode in ('truncate', 'prange'):
                cbo = 'horizontal'
            else:
                cbo = 'vertical'
            if cb_pos is not None:
                cax = fig.add_axes(cb_pos)
                cb = fig.colorbar(cs, cax=cax, orientation=cbo)
            else:
                cb = fig.colorbar(cs, ax=ax, orientation=cbo)
            cb.set_label(sleg)
            if zmode == 't':
                # cb.set_clim(*(cb.get_clim()[::-1]))
                cb.ax.invert_yaxis()
            if zmode == 'p' and zbmode in ('truncate', ):
                cb.ax.set_xlim(np.min(x), np.max(x))
        else:
            cb = None

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        ax.set_ylim(xr)
        ax.set_ylim(yr)

        if label:
            ax.text(
                0.05, 0.95, svar,
                transform=ax.transAxes,
                color = 'w',
                verticalalignment='top',
                horizontalalignment='left',
                fontsize = 18,
                )

        if tight is None:
            tight = False
        if tight:
            fig.tight_layout()

        self.cs = cs
        self.fig = fig
        self.ax = ax
        self.cb = cb

class CorrPlotMult(object):
    def __init__(
            self,
            figsize = (12, 12),
            n=2**10,
            ):
        fig = plt.figure(figsize=figsize)
        w  = .425
        h  = .437
        y2 = .558
        y1 = .07
        x1 = .055
        x2 = .573
        cbo = .01
        cbw = .02
        p1 = CorrPlot(
            mode = 'w',
            ax_pos = [x1, y2, w, h],
            cb_pos = [x1+w+cbo, y2, cbw, h],
            fig = fig,
            n = n,
            draw_cb = True,
            zmode = 't',
            )

        p2 = CorrPlot(
            mode = 'j',
            ax_pos = [x2, y2, w, h],
            fig = fig,
            n = n,
            draw_cb = False,
            zmode = 't',
            )
        p2.ax.yaxis.set_ticklabels([])
        p2.ax.set_ylabel(None)
        p3 = CorrPlot(
            mode = 'w',
            ax_pos = [x1, y1, w, h],
            fig = fig,
            n = n,
            draw_cb = True,
            zbmode = 'prange',
            cb_pos = [x1, y1-cbw-cbo, w, cbw],
            zmode = 'p',
            )
        p4 = CorrPlot(
            mode = 'j',
            ax_pos = [x2, y1, w, h],
            fig = fig,
            n = n,
            draw_cb = True,
            zmode = 'p',
            zbmode = 'prange',
            cb_pos = [x2, y1-cbw-cbo, w, cbw],
            )
        for p in (p3, p4):
            p.ax.xaxis.set_ticks_position("top")
            p.ax.xaxis.set_ticklabels([])
            p.ax.set_xlabel(None)

        self.fig = fig
        self.p = [p1, p2, p3, p4]

    def save(
            self,
            filename = '~/LaTeX/oblique/correlations.png',
            ):
        self.fig.savefig(os.path.expanduser(filename))

# TODO - IMPLEMENT USE OF THIS CODE ELSEWHERE

def calc_NN(
        theta, phi,
        align = True,
        w = [0, 0, 1],
        dw = [0, 0, 1],
        j = None,
        dj = None,
        alignphi = 0,
        n2 = 1,
        k = 2/3, # gyration constant j = w * k * r **2
        ra = 1, # radius
        fsh = 1, # enable N2W term; set to 0 for 'classical' DSI
        ric = 1/4,
        returns = ('n2x', ),
        debug = False,
        ):

    # [BEGIN NX] ---- this part here should is largely a copy ...
    r_hat = hat(theta, phi)
    r = ra * r_hat
    i = k * ra ** 2

    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        j = w * i
    else:
        j = np.asarray(j, dtype=np.float64)
        w = j / i
    if dw is not None:
        dw = np.asarray(dw, dtype=np.float64)
        dj = dw * i + 2 * j / ra
    else:
        dj = np.asarray(dj, dtype=np.float64)
        dw = dj / i - 2 * w / ra

    # print('w',w,'dw',dw,'j',j,'dj',dj)


    mjdj = np.dot( j, dj) / (norm(j)  * norm(dj) + 1.e-199)
    mwdw = np.dot( w, dw) / (norm(w)  * norm(dw) + 1.e-199)
    mg   = np.dot(dw, dj) / (norm(dj) * norm(dw) + 1.e-199)

    # align such that j || z and dj in x-z plane

    assert align
    if align:
        j = np.array([0, 0, 1.]) * norm(j)
        w = np.array([0, 0, 1.]) * norm(w)
        njdj = sqrt(1-mjdj**2)
        dj = np.array([njdj * cos(alignphi),
                       njdj * sin(alignphi),
                       mjdj]) * norm(dj)
        dw = dj / i - 2 * w / ra

    # vector
    jdotrhat = np.tensordot( j, r_hat, (0,0))
    djdotrhat = np.tensordot(dj, r_hat, (0,0))
    jdotdj = np.dot(j, dj)

    nw1 = jdotrhat * djdotrhat
    nw2 = jdotdj
    n2w = 2 * (nw2 - nw1) / (ra**3 * k**2) * fsh

    # DSI
    dwdlr = np.cross(dw, r, axis=0)
    dwdlr2 = np.sum(dwdlr * dwdlr, axis=0)

    # Richardson Criterion
    n2r = - ric * dwdlr2

    # generalised criterion
    n2x = n2 + n2w + n2r

    # diagnostic output
    if 'n2xp' in returns:
        n2xp = np.max(n2x)
    if 'n2xm' in returns:
        n2xm = np.min(n2x)
    if 'fmin' in returns:
        pw = norm(dw) / norm(w)
        fmin = fw_min(pw, mwdw, ric = ric, fsh = fsh)
    # [END NX] ---- this part here should is largely a copy ...

    if debug:
        n2x[:,:] = -0.1**2

    # master code
    results = list()
    if isinstance(returns, str):
        return locals()[returns]
    for r in returns:
        results.append(locals()[r])
    if len(results) == 1:
        return results[0]
    elif len(results) == 0:
        return None
    return results

class Average(object):
    def __init__(
            self, *,
            n = 2**8,
            fig = None,
            ax = None,
            **kwargs,
            ):

        # but we need t set up proper grid for integration
        #
        #
        #    X-------X-------X
        #    |       |       |
        #    |       |       |
        #    |   Z   |   Z   |
        #    |       |       |
        #    |       |       |
        #    X-------X-------X
        #
        #   X: Mesh point for drawing
        #   Z: Staggered location values
        #
        # locations:
        #   f: grid locations (corners of the mesh)

        # set up grid
        thetaf, phif = mgrid[0:pi:(n+1)*1j, 0:2*pi:(n*2+1)*1j]
        self.theta = f2cx(thetaf[:,1:])
        self.phi   = f2cx(phif[:,1:])
        self.n = n
        self.kwargs = kwargs
        self.fig = fig
        self.ax = ax

        #rcParams['text.usetex'] = True
        #rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

    def average(
            self,
            theta, phi,
            returns = ('g',),
            **kwargs,
            ):

        mode = kwargs.pop('mode', 'max') # av, max, int

        n2 = kwargs.setdefault('n2', 1)

        n2x, fmin, w, dw = self.calc_NN(theta, phi, returns = ('n2x', 'fmin', 'w', 'dw',), **kwargs)

        # growth per period
        s = sin(theta[:, 0]) # this could be cached, but cost should be small
        st = np.sum(s)

        if mode in ('av', 'max', 'int', 'pow2', 'pow3', ):
            gt = np.sum(np.sqrt(np.maximum(0,-n2x)), axis=1) / n2x.shape[1]
        elif mode == 'frac':
            gt = np.count_nonzero(n2x < 0, axis=1) / n2x.shape[1]

        if mode in ('av', 'frac', ):
            g = np.sum(gt * s) / st
        elif mode == 'max':
            g = np.max(gt)
        elif mode == 'pow2':
            g = np.sqrt(np.sum(gt**2 * s) / st)
        elif mode == 'pow3':
            g = (np.sum(gt**3 * s) / st)**(1/3)
        elif mode == 'int':
            # do quad precision calc (numpy 1.17: only native 10 byte floats)
            gt = np.float128(gt)
            g = np.log(np.sum(np.exp(gt) * s) / st)
            g = np.float64(g)
            # there seems to be some issue converting back that I have not understood
            if g < 1e-15:
                g *= 0

        w2 =  np.dot(w, w)
        n2xmin = n2 + w2 * fmin
        if mode in ('frac', ):
            gn = g
        else:
            gn = g / (sqrt(max(1.e-99, -n2xmin)))

        debug = kwargs.get('debug', False)
        if debug:
            gn = g

        # if g > 10:
        #     breakpoint()

        # master code
        results = list()
        if isinstance(returns, str):
            return locals()[returns]
        for r in returns:
            results.append(locals()[r])
        if len(results) == 1:
            return results[0]
        elif len(results) == 0:
            return None
        return results


    def plot0(self):

        g, gt, w, dw = self.average(
            self.theta,
            self.phi,
            returns = ('g', 'gt', 'w', 'dw', ),
            **self.kwargs)

        fig = plt.figure()

        ax = fig.add_subplot(2,1,1)
        ax.plot(self.theta[:,0], gt)
        ax.axhline(g, ls=':')
        ax.set_ylabel('average growth\nper revolution')
        ax.set_xlabel(r'$\theta$')
        ax.margins(x=0)

        ax = fig.add_subplot(2,1,2)
        ax.plot(self.theta[:,0], np.exp(gt))
        ax.axhline(np.exp(g), ls=':')
        ax.set_ylabel('average growth factor\nper revolution')
        ax.set_xlabel(r'$\theta$')
        ax.margins(x=0)
        ax.text(0.95,0.95, 'dw='+str(dw)+'\nw='+str(w),
                transform=ax.transAxes,
                va='right', ha='top',
                )
        fig.tight_layout()

    def plot1(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        kwargs = self.kwargs.copy()
        nn2 = kwargs.pop('nn2', 10)
        n2_max = kwargs.pop('n2', 10)

        n2 = mgrid[0:n2_max:1j*nn2]

        a = []
        for nx in n2:
            kwargs['n2'] = nx
            g, gt, w, dw = self.average(
                self.theta,
                self.phi,
                returns = ('g', 'gt', 'w', 'dw', ),
                **kwargs)
            a.append(g)
        ax.plot(n2, a)
        ax.margins(x=0)
        ax.set_xlabel(r'$N^2$')
        ax.set_ylabel(r'$\left<\sqrt{-f}\right>/\sqrt{-f_{\min}}$')
        ax.text(0.95,0.95, 'dw='+str(dw)+'\nw='+str(w),
                transform=ax.transAxes,
                ha='right', va='top',
                )
        fig.tight_layout()

    def get_slab(self, mode = 'av', ric = 1/4, fsh = 1):
        sifun = ''
        snn = fr'-N^2_{{\mathrm{{X}}}}'
        if ric == 0:
            sifun = '\Omega,'
            snn = fr'-N^2-N^2_{{\Omega}}'
        if fsh == 0:
            sifun = fr'\mathrm{{R}},'
            snn = fr'-N^2-N^2_{{\mathrm{{R}}}}'

        if mode in ('int', ):
            slab = fr'$\ln\left<\exp\left\{{\frac{{1}}{{2\pi}}\int_{{0}}^{{2\pi}}\sqrt{{{snn}}}\,\mathrm{{d}}\varphi\right\}}\right>\,\left/\,\sqrt{{-N^2-\Omega^2\,f_{{{sifun}\min}}}}\right.$'
            # slab = fr'$\ln\left<\exp\left\{{\mathrm{{latitudinal\;average\;of}}\;\sqrt{{{snn}}}\right\}}\right>\,\left/\,\sqrt{{-N^2-\Omega^2\,f_{{{sifun}\min}}}}\right.$'
        elif mode in ('av', ):
            slab = fr'$\left<\sqrt{{{snn}}}\right>\,\left/\,\sqrt{{-N^2-\Omega^2\,f_{{{sifun}\min}}}}\right.$'
        elif mode in ('pow2', ):
            slab = fr'$\left<\left(\frac{{1}}{{2\pi}}\int_{{0}}^{{2\pi}}\sqrt{{{snn}}}\,\mathrm{{d}}\varphi\right)^2\right>^{{1/2}}\,\left/\,\sqrt{{-N^2-\Omega^2\,f_{{{sifun}\min}}}}\right.$'
        elif mode in ('pow3', ):
            slab = fr'$\left<\left(\frac{{1}}{{2\pi}}\int_{{0}}^{{2\pi}}\sqrt{{{snn}}}\,\mathrm{{d}}\varphi\right)^3\right>^{{1/3}}\,\left/\,\sqrt{{-N^2-\Omega^2\,f_{{{sifun}\min}}}}\right.$'
        elif mode in ('max', ):
            # slab = fr'[maximum of average along $\theta$]$\left(\sqrt{{{snn}}}\right)\,\left/\,\sqrt{{-N^2-\Omega^2\,f_{{{sifun}\min}}}}\right.$'
            # slab = fr'$\left[\mathrm{{maximum\;of\;longitudinal\;average}}\right]\left(\sqrt{{{snn}}}\right)\,\left/\,\sqrt{{-N^2-\Omega^2\,f_{{{sifun}\min}}}}\right.$'
            slab = fr'$\max\left(\frac{{1}}{{2\pi}}\int_{{0}}^{{2\pi}}\sqrt{{{snn}}}\,\mathrm{{d}}\varphi\right)\,\left/\,\sqrt{{-N^2-\Omega^2\,f_{{{sifun}\min}}}}\right.$'
        elif mode in ('frac', ):
            snn = snn.replace('-','+').lstrip('+')
            slab = fr'fraction of surface with $\;{snn}\,<0$'

        return slab

    def plot2(self, **_kwargs):

        kwargs = self.kwargs.copy()
        kwargs.update(_kwargs)

        ric = self.kwargs.setdefault('ric', 1/4)
        fsh = self.kwargs.setdefault('fsh', 1)

        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = plt.figure()
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = fig.add_subplot(1,1,1)

        nn = kwargs.pop('nn', 10)

        pwf, mwf = mgrid[-2.5:+3:1j*(nn+1), -1: +1: 1j*(nn+1)]

        pw = 10**f2c(pwf)
        mw = f2c(mwf)

        w = np.array([0, 0, 1.])

        f = np.empty_like(pw)

        kwargs['w'] = w

        mode = kwargs.setdefault('mode', 'max') # av, max, int

        slab = self.get_slab(mode = mode, ric = ric, fsh = fsh)

        n2 = kwargs.setdefault('n2', 1)

        for (i,j) in itertools.product(range(f.shape[0]), range(f.shape[1])):
                dw = np.array([np.sqrt(1-mw[i,j]**2), 0., mw[i,j]]) * pw[i,j]
                kwargs['dw'] = dw
                gn = self.average(
                    self.theta,
                    self.phi,
                    returns = 'gn',
                    **kwargs)
                f[i,j] = gn

        # print(np.max(f))

        if n2 > 0:
            snval = '{:-6.2f}'.format(np.log10(np.maximum(n2, 1.e-99)).tolist())
            stit = fr'$\log\left(N^2/\,\Omega^2\right)={snval}$'
        elif n2 < 0:
            snval = '{:-6.2f}'.format(np.log10(np.maximum(-n2, 1.e-99)).tolist())
            stit = fr'$\log\left(-N^2/\,\Omega^2\right)={snval}$'
        else:
            stit = fr'$N^2/\,\Omega^2=0$'

        ax.text(0.05, 0.95, stit,
                transform=ax.transAxes,
                ha='left', va='top',
                color = 'white',
                fontsize = 12,
                )
        ax.set_xlabel(r'$\log\,t$')
        ax.set_ylabel(r'$\breve{\mu}$')

        cm = 'plasma'
        cm = get_cmap(cm)
        cm.set_bad('#000000')

        vmin=0
        vmax=1

        f[f==0] = np.nan

        cs = ax.pcolormesh(pwf, mwf, f, vmin=vmin, vmax=vmax, cmap = cm)
        cb = fig.colorbar(cs, ax=ax, label = slab)

        fig.tight_layout()
        self.fig = fig
        self.ax = ax
        self.cb = cb
        self.cs = cs

    @classmethod
    def movie2(
            cls,
            n2min = -5,
            n2max = +5,
            n2del = 0.02,
            size = (800, 600),
            dpi = 100,
            filename = '~/Downloads/growth2.webm',
            parallel = False,
            framerate = 60,
            **kwargs,
            ):
        nframes = round((n2max - n2min) / n2del) + 1
        kwargs.setdefault('n', 2**6)
        kwargs.setdefault('nn', 2**6)
        MovieWriter.make(
            filename,
            base = cls,
            bkwargs = dict(**kwargs),
            func = 'plot2',
            data = 'n2',
            values = 10**mgrid[n2min:n2max:1j*(nframes)],
            canvas = MPLBase,
            ckwargs = dict(size=size, dpi=dpi),
            framerate = framerate,
            parallel = parallel,
            )
# A.movie2(n=2**6, nn=2**8)

    def plot3(self, **_kwargs):
        # boundary plot

        kwargs = self.kwargs.copy()
        kwargs.update(_kwargs)

        ric = self.kwargs.setdefault('ric', 1/4)
        fsh = self.kwargs.setdefault('fsh', 1)

        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = plt.figure()
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = fig.add_subplot(1,1,1)

        nn = kwargs.pop('nn', 20)

        n2min = self.kwargs.setdefault('n2min', -5.0)
        n2max = self.kwargs.setdefault('n2max', +5.0)
        pwmin = self.kwargs.setdefault('pwmin', -4.0)
        pwmax = self.kwargs.setdefault('pwmax', +4.0)

        pwf, n2f = mgrid[pwmin:pwmax:1j*(nn+1), n2min:n2max:1j*(nn+1)]

        pwf = np.append(pwf[::-1,:], pwf, axis=0)
        n2f = np.append(n2f, n2f, axis=0)

        pwf = 10**pwf
        pwf[:pwf.shape[0]//2,:] *= -1
        pw = f2c(pwf)

        n2 = 10**f2c(n2f)

        w = np.array([0, 0, 1.])

        f = np.empty_like(pw)

        kwargs['w'] = w

        mode = kwargs.setdefault('mode', 'max') # av, max, int

        slab = self.get_slab(mode = mode, ric = ric, fsh = fsh)

        for (i,j) in itertools.product(range(f.shape[0]), range(f.shape[1])):
                dw = w * pw[i,j]
                kwargs['dw'] = dw
                kwargs['n2'] = n2[i,j]
                gn = self.average(
                    self.theta,
                    self.phi,
                    returns = 'gn',
                    **kwargs)
                f[i,j] = gn

        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\log\;N^2$')
        loglim = 10**(pwmin+2)
        ax.set_xscale('symlog',linthreshx = loglim)

        cm = 'plasma'
        cm = get_cmap(cm)
        cm.set_bad('#000000')

        vmin=0
        vmax=1

        f[f==0] = np.nan

        cs = ax.pcolormesh(pwf, n2f, f, vmin=vmin, vmax=vmax, cmap = cm)
        cb = fig.colorbar(cs, ax=ax, label = slab)

        fig.tight_layout()
        self.fig = fig
        self.ax = ax
        self.cb = cb
        self.cs = cs

class GrowthPlot(object):
    def __init__(
            self,
            n2 = 0,
            w = 1,
            xmag = 3,
            xmin = -2.5, # -2.5
            xmax = +3.0, # +3.0
            xrange = None,
            n = 2**10,
            ric = 1/4,
            fsh = 1,
            ymode = 'theta', # theta | mu
            ymin = None,
            ymax = None,
            yrange = None,
            xmode = 'w', # j | w
            fig = None,
            ax = None,
            figsize = None,
            dpi = None,
            ):

        if xrange is not None:
            x0, x1 = xrange
        else:
            x0, x1 = -xmag, xmag
        if xmin is not None:
            x0 = xmin
        if xmax is not None:
            x1 = xmax

        if yrange is not None:
            y0, y1 = yrange
        else:
            if ymode == 'theta':
                y0, y1 = 0, pi
            else:
                y0, y1 = 1, -1
        if ymode == 'mu':
            ymin, ymax = ymax, ymin
        if ymin is not None:
            y0 = ymin
        if ymax is not None:
            y1 = ymax

        ########## the rest of this is work in progress

        xf, yf = np.mgrid[x0:x1:1j*(2*n+1), y0:y1:1j*(n+1)]

        xc = f2c(xf)
        yc = f2c(yf)

        # best after centering
        xc  = 10**xc

        if ymode == 'theta':
            mu = cos(yc)
        else:
            mu = yc

        p = xc

        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        if ax is None:
            ax = fig.add_subplot(1,1,1)

        vmin = None
        vmax = None

        xr = None
        yr = None

        if ric == 0:
            smode = r'{{\Omega}}'
            sfmode = fr'_{smode}'
            stmode = fr'{smode},'
        elif fsh == 0:
            smode = r'{{\mathrm{{R}}}}'
            sfmode = fr'_{smode}'
            stmode = fr'{smode},'
        else:
            smode = ''
            stmode = ''
            sfmode = ''

        if xmode == 'w':
            fn = fw_min
            sfun = f'f'
            spvar = fr'\breve\theta'
            smvar = fr'\breve\mu'
            sxvar = f't'
        else:
            spvar = fr'\tilde\theta'
            smvar = fr'\tilde\mu'
            sxvar = f's'
            fn = fj_min
            sfun = f'g'

        xlab = rf'$\log\,{sxvar}$'
        x = xf

        if ymode == 'theta':
            ylab = fr'${spvar}$'
            y = yf * (180 / pi)
            yr = np.array([y1, y0]) * (180 / pi)
            ax.axhline(90, ls='--', color='k')
        else:
            ylab = fr'${smvar}$'
            y = yf
            yr = [y1, y0]
            ax.axhline(0, ls='--', color='k')

        f = fn(p, mu, ric = ric, fsh = fsh)
        n2x = n2 + w**2 * f
        z = np.log10(sqrt(np.maximum(-n2x, 1.e-99)))
        z[z<-10] = np.nan

        sleg = fr'$\log\,\sqrt{{-N^2-\Omega^2\,{sfun}_{{{stmode}\min}}}}$'

        cm = 'plasma'
        cm = ColorBlendWave(
            cm,
            'black',
            amplitude=0.02,
            power=0,
            nwaves=25,
            )
        cm.set_bad('#000000')

        if vmin is None:
            vmin = x[0,0]*2+0.1
        if vmax is None:
            vmax = x[-1,-1]-0.3

        cs = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap = cm)
        # sleg += '\n' + r'$\longleftarrow$ more unstable'

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        ax.set_ylim(xr)
        ax.set_ylim(yr)

        cbo = 'vertical'
        cb = fig.colorbar(cs, ax=ax, orientation=cbo)
        cb.set_label(sleg)

        if n2 > 0:
            snval = '{:+6.2f}'.format(np.log10(np.maximum(n2, 1.e-99)).tolist())
            stit = fr'$\log\left(N^2/\,\Omega^2\right)={snval}$'
        elif n2 < 0:
            snval = '{:+6.2f}'.format(np.log10(np.maximum(-n2, 1.e-99)).tolist())
            stit = fr'$\log\left(-N^2/\,\Omega^2\right)={snval}$'
        else:
            stit = fr'$N^2/\,\Omega^2=0$'
        ax.text(0.05, 0.95, stit,
                transform=ax.transAxes,
                ha='left', va='top',
                color = 'white',
                fontsize = 12,
                )

        fig.tight_layout()

        self.fig = fig
        self.ax = ax

    @classmethod
    def movie(
            cls,
            n2min = -5,
            n2max = +5,
            n2del = 0.02,
            size = (800, 600),
            dpi = 100,
            filename = '~/Downloads/growth_rate_t.mp4',
            framerate = 60,
            **kwargs,
            ):
        nframes = round((n2max - n2min) / n2del) + 1
        kwargs.setdefault('n', 2**10)
        ParallelMovie.make(
            filename,
            fkwargs = dict(**kwargs),
            func = cls,
            data = 'n2',
            values = 10**mgrid[n2min:n2max:1j*(nframes)],
            canvas = MPLBase,
            ckwargs = dict(size=size, dpi=dpi),
            framerate = framerate,
            )
# G.movie(n=2**10,ymode='theta',xmode='j', filename = '~/Downloads/growth_rate_j.webm')
# G.movie(n=2**10,ymode='theta',xmode='w', filename = '~/Downloads/growth_rate_w.webm')



examples = """
# Some movie examples

from dsiplot import Movie as M
M(filename = '~/Downloads/DSI.webm', mode = 'shade')
M(filename = '~/Downloads/DSI_contour.webm', mode = 'contour')
M(filename = '~/Downloads/DSI_project.webm', mode = 'project')

from dsiplot import MovieParallel as P
P(filename = '~/Downloads/DSI_project.webm', mode = 'project', vmin=-6.5, vmax=2.5)
P(filename = '~/Downloads/DSI.webm', mode = 'shade', mag=6.5)
P(filename = '~/Downloads/DSI_contour.webm', mode = 'contour')
P(filename = '~/Downloads/DSI_min.webm', mode = 'min', vmin =-6.5, vmax=2.5)
P(filename = '~/Downloads/DSI_sphere.webm', mode = 'sphere', mag=6.5)

from dsiplot import MaxPlot as M
M(mode = 'value', nt = 2**8, n=2**8)
M(mode = 'angle', nt = 2**6, n=2**10)
M(mode = 'average', nt = 2**10, n=2**10)
M(mode = '2D', nt = 2**8, n=2**10)

M(mode = '2D', nt = 2**8, n=2**8, n2=1, fw=0, ra=1, dja=1)
"""

"""
# complex max test
from dsiplot import test_solution as T
T(1)
"""

"""
# minplot
from dsiplot import MinPlot as M

M(mode='theta')
M()
M(trange=[0.982,0.986], mrange=[0.99975,1])
M(mode=None, trange=[0.95,1.5], mrange=[0.96,1])
M(mode=None, trange=[0.2,.35], mrange=[-1,-0.99])
"""
