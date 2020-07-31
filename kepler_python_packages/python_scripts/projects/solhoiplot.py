import numpy as np
from numpy import mgrid
import matplotlib.pyplot as plt
from numpy.linalg import norm

try:
    from mpl_toolkits.basemap import Basemap
except:
    print('Could not import basemap.')

from rotation import rotscale, rotate2, w2p

from movie import MovieWriter
from movie.mplbase import MPLBase
from movie import ParallelMovieFrames, ParallelMovie

pi=np.pi
sin=np.sin
cos=np.cos
sqrt = np.sqrt
r2d = 180/pi

class SolHoiPlot(object):

    @staticmethod
    def hat(theta, phi):
        s = sin(theta)
        x = s * cos(phi)
        y = s * sin(phi)
        z = cos(theta)
        return np.array([x,y,z])

    def __init__(
            self, *,
            finite = False,
            normalize = True,
            align = None,
            j = [0,0,1],
            dj = [1,0,0],
            n = 2**10,
            mode = 'shade',
            fig = None,
            ax = None,
            alignphi = 0,
            proj = 'ortho',
            N2=1,
            cmap = 'bwr_r'
            ):

        if align is None:
            if mode == 'sphere':
                align = False
            else:
                align = True

        theta, phi = mgrid[0:pi:n*1j, 0:2*pi:n*2j]
        r_hat = self.hat(theta,phi)

        dj = np.asarray(dj, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)

        if finite:
            jp = j + 0.5 * dj
            jm  = j - 0.5 * dj

            # the rest of the code in this block should ideally be
            # used for numerical derivatives in applications such as
            # KEPLER
            j = rotscale(jp,jm)
            dj = jp - jm

            # do correction for proper analytic formula (...)
            rot2 = rotate2(jp, jm)
            dj = np.cross(j, rot2) \
                 + j / norm(j) \
                 * (norm(jp) - norm(jm))

        snorm0 = r'\left|\vec{{j}}\right|\cdot\left|\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right|'
        snorm=''
        if normalize:
            j  /= norm( j)
            dj /= norm(dj)
            snorm = rf'\,\left/\,\left({snorm0}\right)\right.'

        # align such that j || z and dj in x-z plane
        if align:
            mu = np.dot(j, dj) / (norm(j) * norm(dj))
            j = np.array([0, 0, 1.]) * norm(j)
            dj = np.array([sqrt(1-mu**2), alignphi, mu]) * norm(dj)

        muj = np.dot(j, dj) / (norm(j) * norm(dj))

        # vector
        jdotrhat = np.tensordot( j, r_hat, (0,0))
        djdotrhat = np.tensordot(dj, r_hat, (0,0))
        jdotdj = np.dot(j, dj)

        mu = cos(theta)
        p = phi/(2*pi)

        nw1 = jdotrhat * djdotrhat
        nw2 = jdotdj
        n2w = 2 * (nw2 - nw1)

        n2wp = np.max(n2w)
        n2wm = np.min(n2w)

        if mode == 'calc':
            self.n2wp = n2wp
            self.n2wm = n2wm

            ii = np.argmin(n2w)
            jt = ii // n2w.shape[1]
            jp = ii - jt * n2w.shape[1]

            self.n2wmt = theta[jt, jp]
            self.n2wmp = phi[jt ,jp]

            self.n2wam = np.min(np.average(n2w, axis=1))

            ii = np.argmin(np.average(n2w, axis=1))
            self.n2wamt = theta[ii, 0]

            self.n2wtm = np.min(n2w, axis=1)
            self.t = theta[:,0]

            return

        print(f'cos(j,dj)={muj:5.3f}')

        # diagnotic output


        print(f'[LOCAL] maximum = {n2wp:4.2f}, minimum = {n2wm:4.2f}')

        scos = rf'\cos\left(\vec{{j}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right)={muj:+4.2f}'
        sdj = rf'\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\left[\,\vec{{j}}\times\left(\vec{{j}}\times\hat{{r}}\right)\right]\cdot\hat{{r}}'
        sdjn = rf'{sdj}{snorm}'


        jpol=w2p(j)
        djpol=w2p(dj)

        if fig is None:
            fig, ax = plt.subplots()
        if ax is None:
            ax = fig.add_subplot(111)

        sleg = r'less stable $\longleftrightarrow$ more stable'

        mag = 2
        if not normalize:
            mag *= norm(j) * norm(dj)
        vmin = -mag
        vmax = +mag

        if mode in ('shade', 'sphere', ):
            n2wc = 0.25 * (n2w[:-1,:-1] + n2w[1:,:-1] + n2w[:-1,1:] + n2w[1:,1:])

        if mode == 'contour':
            cs = ax.contour(p, mu, n2w)
            ax.clabel(cs, inline=True, fontsize=8)
        elif mode == 'shade':
            cs = ax.pcolormesh(p, mu, n2wc, vmin=vmin, vmax=vmax, cmap=cmap)
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
                cs = m.pcolor(lon, lat, n2wc, vmin=vmin, vmax=vmax, cmap=cmap, latlon=True)
            else:
                # fix for pcolormesh (created pul request)
                x,y = m(lon,lat)
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
                n2wc[:nx-1,:ny-1][ii]=np.nan
                cs = m.pcolormesh(lon, lat, n2wc, vmin=vmin, vmax=vmax, cmap=cmap, latlon=True)
            leg_loc = 'lower left'

            stit = f'${sdjn}$'
            fig.text(0.01, 0.99, stit, va='top', fontsize='large', color='k')


        elif mode in ('project', 'min'):
            if mode == 'project':
                n2wa = np.average(n2w, axis=1)
                sxlab = rf'$\left<\,{sdjn}\,\right>$'
            else:
                n2wa = np.min(n2w, axis=1)
                sxlab = rf'$\min\left(\,{sdjn}\,\right)$'
            ax.plot(n2wa, mu[:,0])

            n2wap = np.max(n2wa)
            n2wam = np.min(n2wa)

            print(f'[AVERAGES] maximum = {n2wap:4.2f}, minimum = {n2wam:4.2f}')

            leg_loc = 'upper right'
            ax.set_ylabel(r'$\mu$')
            ax.set_xlabel(sxlab)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(-1, 1)

            djpol[2] = jpol[2] = 0

        if mode in ('shade', 'sphere'):
            cb = fig.colorbar(cs, ax=ax)
            cb.set_label(sleg)

        if mode in ('contour', 'shade', ):
            stit = fr'${sdjn},\quad{scos}$'
            ax.set_title(stit)

            ax.set_xlabel(r'Phase')
            ax.set_ylabel(r'$\mu$')
            leg_loc = 'lower right'

        x,y = np.array([[jpol[2], djpol[2]],
                        [jpol[1], djpol[1]]])
        if mode == 'sphere':
            x, y = m(x * r2d, 90 - y * r2d)
            x[x > 1e20] = np.nan
            y[y > 1e20] = np.nan
        else:
            x /= 2*pi
            y = cos(y)

        ax.plot(x[:1], y[:1], ls='none', marker='o', markersize=8, color="#00af7f", clip_on=False, label = r'$\vec{j}\,/\,\left|\vec{j}\right|$')
        ax.plot(x[1:], y[1:], ls='none', marker='^', markersize=8, color="#ffaf00", clip_on=False, label = r'$\frac{\mathrm{d}}{\mathrm{d}r}\vec{j}\,/\,\left|\frac{\mathrm{d}}{\mathrm{d}r}\vec{j}\right|$')
        leg = fig.legend(loc=leg_loc, framealpha = 1)

        if mode in ('project', 'min' ):
            l = ax.axvline(-N2, c='r', ls=':',
                       label = f'stability limit for\n$N^2\,k^2\,r^3={snorm0}$')
            ii = np.where(n2wa > -N2)[0]
            x = n2wa
            y = mu[:,0]
            y[ii] = np.nan
            ax.fill_betweenx(y,x,-N2, color='#ffbbbb')
            ax.legend(handles=[l], loc='lower right', framealpha = 1)

        fig.tight_layout()

        self.fig = fig
        self.ax = ax


def adaptive_curve(x0, x1, n0, f, eps = 1.e-8):
    x = np.mgrid[x0:x1:n0*1j].tolist()
    y = [f(xi) for xi in x]

    def metric(x0,x1,x2,y0,y1,y2):
        vx = (x2-x0)
        vy = (y2-y0)
        m = np.sqrt(vx**2 + vy**2)
        nx = -vy / m
        ny = +vx / m
        d2 = x0 * nx + y0 * ny
        d1 = x1 * nx + y1 * ny
        dd = d2 - d1
        tx = dd * nx
        ty = dd * ny
        # possible norms
        na = np.abs(dd) / m
        nb = np.sqrt(tx**2/(vx**2 + 1.e-99) + ty**2/(vy**2 + 1.e-99))
        if nb >  1e10:
            raise Exception('This should not occur.')
        return nb

    i = 0
    while i < len(x) - 2:
        m = metric(*x[i:i+3], *y[i:i+3])
        if m > eps:
            x2 = 0.5 * (x[i] + x[i+1])
            y2 = f(x2)
            x.insert(i+1, x2)
            y.insert(i+1, y2)
            print(i,x2,y2,m)
        else:
            i += 1
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y


class MaxPlot(object):
    def __init__(
            self,
            n = 2**8,
            nt = 2**8,
            mode = 'angle',
            ):

        j = [0,0,1]

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
        for cx,sx in zip(c,s):
            s = SolHoiPlot(
                mode = 'calc',
                j=j,
                dj=[sx,0,cx],
                n = n)
            m.append(s.n2wm)
            a.append(s.n2wam)
            at.append(s.n2wamt)
            mt.append(s.n2wmt)
            mp.append(s.n2wmp)

            tx.append(s.t)
            mx.append(s.n2wtm)

        mt = np.asarray(mt)
        mp = np.asarray(mp)
        at = np.asarray(at)
        a  = np.asarray(a)
        m  = np.asarray(m)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        smuj=rf'$\mu=\cos\left(\vec{{j}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right)$'
        saj=rf'$\sphericalangle\left(\vec{{j}},\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right)$'
        snorm0 = r'\left|\vec{{j}}\right|\cdot\left|\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\,\vec{{j}}\right|'
        snorm = rf'\,\left/\,\left({snorm0}\right)\right.'
        sdj = rf'\frac{{\mathrm{{d}}}}{{\mathrm{{d}}r}}\left[\,\vec{{j}}\times\left(\vec{{j}}\times\hat{{r}}\right)\right]\cdot\hat{{r}}'
        sdjn = rf'{sdj}{snorm}'
        sdjmin = rf'$\min\left(\,{sdjn}\,\right)$'
        sdjmina = rf'location of $\min\left(\,{sdjn}\,\right)$'
        st = rf'$\theta=\sphericalangle\left(\vec{{j}},\vec{{r}}\right)$'

        if mode == 'angle':
            ax.plot(t * r2d, mt * r2d, color = 'k', ls = '-', label = r'$\theta$')
            ax.plot(t * r2d, mp * r2d, color = 'k', ls = ':', label = r'$\phi$')
            ax.set_ylabel(sdjmina)
            ax.set_xlabel(saj)
        elif mode == 'value':
            ax.plot(c, m, color = 'k', label = 'absolute minimum')
            ax.plot(c, a, color = 'k', ls = ':', label = 'min. of lat. average')
            ax.set_ylabel(sdjmin)
            ax.set_xlabel(smuj)
        elif mode == 'average':
            ax.plot(t * r2d, at * r2d, color = 'k', ls = ':',  label = r'$\theta$')
        elif mode == '2D':
            mx = np.asarray(mx).transpose()
            cmx = cos(mx)
            tx = np.asarray(tx[0])
            ctx = cos(tx)
            cs = ax.pcolormesh(t*r2d, tx*r2d, mx, vmin=-2,vmax=2, cmap = 'bwr_r')
            cb = fig.colorbar(cs, ax=ax)
            ax.set_xlabel(saj)
            ax.set_ylabel(st)
            cb.set_label(rf'minimum of ${sdjn}$ on parallels at $\theta$')
            ax.plot(t * r2d, mt * r2d, color = 'k', ls = '-', label = 'absolute\nminimum')
            ax.plot(t * r2d, (pi-mt) * r2d, color = 'k', ls = '-')


        leg = ax.legend(loc='best', framealpha = 1)
        fig.tight_layout()
        self.fig = fig
        self.ax = ax

class Movie(object):
    def __init__(
            self,
            nframes = 180,
            framerate = 30,
            size = (800, 600),
            filename = '~/Downloads/SHI.webm',
            mode = 'contour',
            dpi = 100,
            fig = None,
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
            )
        j = [0,0,1]

        for i,t in writer.enumerate(theta):
            print(f'Making frame {i+1} of {len(theta)}.')
            dj = -np.array([sin(t),0,cos(t)])
            fig.clear()
            SolHoiPlot(dj=dj, j=j, fig = fig, mode = mode)

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
        SolHoiPlot(dj=dj, j=j, fig = self.fig, **self.kwargs)
        return self.mfig.get_frame()

    def close(self):
        self.mfig.close()

class MovieParallel(object):
    def __init__(
            self,
            nframes = 720,
            framerate = 60,
            size = (800, 600),
            filename = '~/Downloads/SHI.webm',
            nparallel = None,
            dpi = 100,
            **kwargs,
             ):

        # this is if we do a loop - exclude otherwise duplicate frame
        theta = 2 * pi * np.arange(nframes)/nframes

        print('Starting.')

        p = ParallelMovie(
            filename,
            nparallel = nparallel,
            delay = 1/framerate,
            generator = FrameGenerator,
            gargs = (size, dpi),
            gkwargs = kwargs,
            )
        p.run(theta)

        print('Done.')

examples = """
# Some movie examples

from solhoiplot import Movie as M
M(filename = '~/Downloads/SHI.webm', mode = 'shade')
M(filename = '~/Downloads/SHI_contour.webm', mode = 'contour')
M(filename = '~/Downloads/SHI_project.webm', mode = 'project')

from solhoiplot import MovieParallel as P
P(filename = '~/Downloads/SHI_project.webm', mode = 'project')
P(filename = '~/Downloads/SHI.webm', mode = 'shade')
P(filename = '~/Downloads/SHI_contour.webm', mode = 'contour')
P(filename = '~/Downloads/SHI_min.webm', mode = 'min')
P(filename = '~/Downloads/SHI_sphere.webm', mode = 'sphere')

from solhoiplot import MaxPlot as M
M(mode = 'value', nt = 2**8, n=2**8)
M(mode = 'angle', nt = 2**6, n=2**10)
M(mode = 'average', nt = 2**10, n=2**10)
M(mode = '2D', nt = 2**8, n=2**10)
"""
