import convdata
import convplot
import np
import matplotlib.pylab as plt
import os.path


class KH_nice():
    def __init__(self, conv = '~/kepler/prog3d/s15/s15.cnv'):
        if isinstance (conv, str):
            conv = convdata.load(conv)
        self.conv = conv
    def plot(self):
        self.p = convplot.plot(self.conv, logtime = -8.2, solar = True, full_conv_names = True)
    def labels(self):
        ax = self.p.ax
        ax.texts = []
        parm = dict(verticalalignment='center',
                    horizontalalignment='center',
                    fontsize=10)
        ax.text(6.8, 0.5, 'H', **parm)
        ax.text(5.3, 0.5, 'He', **parm)
        ax.text(5.3, 4.4, 'H shell', **parm)
        ax.text(3.6, 2.8, 'He shell', **parm)
        ax.text(3, 0.5, 'C', **parm)
        ax.text(2.5, 0.9, 'C', **parm)
        ax.text(1.9, 1.4, 'C', **parm)
        ax.text(1.2, 1.9, 'C', **parm)
        ax.text(1, 0.3, 'Ne', **parm)
        ax.text(0, 0.3, 'O', **parm)
        ax.text(-0.4, 1.2, 'Ne', **parm)
        ax.text(-1, 1.1, 'O', **parm)
        ax.text(-2, 1.7, 'O', **parm)
        ax.text(-2.5, 0.3, 'Si', **parm)
        ax.text(-6, 1.0, 'Si', **parm)
        ax.text(-6, 2.3, 'C', **parm)
        ax.text(-2, 8, 'Red Super Giant Envelope', **parm)
        ax.text(5.2, 13.1, 'Mass Loss', rotation = 300, **parm)

class MultiKHD():
    def __init__(self):
        masses = [9.5, 12, 15, 20, 25, 40]
        path0 = '/home/alex/kepler/prog3d/'
        sentinel = 's'
        files = [os.path.join(
            path0,
            sentinel + str(mass),
            sentinel + str(mass) + '.cnv.xz')
                 for mass in masses]
        self.cnv = [convdata.load(f) for f in files]
        self.labels = [r'${}\,\mathrm{{M}}_\odot$'.format(mass)
                       for mass in masses]
    def plot(self, figsize = (8, 12)):
        self.f = plt.figure(
            dpi = 102,
            figsize = figsize,
            )
        self.axes = []
        for i,c in enumerate(self.cnv):
            # c = self.cnv[-3]
            ax = self.f.add_subplot(321 + i)
            self.axes.append(ax)
            convplot.plot(
                c,
                axes = ax,
                showlegend = False,
                showconvlegend = False,
                logtime = -8.2,
                solar = True,
                auto_position = False,
                interactive = False,
                )
            ax.text(0.95, 0.95, self.labels[i],
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)
        self.f.tight_layout()
        basename = '/home/alex/a/Downloads/KHDm{}x{}'.format(*figsize)
        self.f.savefig(basename + '.png')
        self.f.savefig(basename + '.pdf')

class MultiKHDr():
    def __init__(self):
        mass = '30'
        path0 = '/g/alex/kepler/gridb/'
        rotations = ['0000','0250','0500','0750','1000','1500']
        sentinel = 'o'
        files = [os.path.join(
            path0,
            sentinel + 'gridb' + r + '-100',
            sentinel + str(mass),
            sentinel + str(mass) + '.cnv')
                 for r in rotations]
        self.labels = [r'$\omega/\omega_\mathrm{{crit}}{:2d}\,\%$'.format(int(int(r) / 30))
                       for r in rotations]
        self.cnv = [convdata.load(f) for f in files]
        self.masses = masses
    def plot(self, figsize = (8, 12)):
        self.f = plt.figure(
            dpi = 102,
            figsize = figsize,
            )
        self.axes = []
        for i,c in enumerate(self.cnv):
            # c = self.cnv[-3]
            ax = self.f.add_subplot(321 + i)
            self.axes.append(ax)
            convplot.plot(
                c,
                axes = ax,
                showlegend = False,
                showconvlegend = False,
                logtime = -8.2,
                solar = True,
                auto_position = False,
                interactive = False,
                )
            ax.text(0.95, 0.99, self.labels[i],
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)
        self.f.tight_layout()
        basename = '/home/alex/a/Downloads/KHDr{}x{}'.format(*figsize)
        self.f.savefig(basename + '.png')
        self.f.savefig(basename + '.pdf')

class MultiKHDz():
    def __init__(self):
        mass = '30'
        path0 = '/m/chris/zdep'
        metals = ['02', '00', '-02', '-04','-10','-40']
        sentinels = [m.replace('-', '_') for m in metals]
        files = [os.path.join(
            path0,
            '4mass' + mass,
            'sin'+ sentinel,
            'sin.cnv')
                 for sentinel in sentinels]
        self.labels = [r'$\log(Z/\mathrm{{Z}}_\odot) = {:+3.1f}$'.format(int(z) / 10)
                       for z in metals]
        self.cnv = [convdata.load(f, fix_surf_10500 = True) for f in files]
        self.masses = masses
    def plot(self, figsize = (8, 12)):
        self.f = plt.figure(
            dpi = 102,
            figsize = figsize,
            )
        self.axes = []
        for i,c in enumerate(self.cnv):
            # c = self.cnv[-3]
            ax = self.f.add_subplot(321 + i)
            self.axes.append(ax)
            convplot.plot(
                c,
                axes = ax,
                showlegend = False,
                showconvlegend = False,
                logtime = -8.2,
                solar = True,
                auto_position = False,
                interactive = False,
                )
            ax.text(0.95, 0.99, self.labels[i],
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)
        self.f.tight_layout()
        basename = '/home/alex/a/Downloads/KHDz{}x{}'.format(*figsize)
        self.f.savefig(basename + '.png')
        self.f.savefig(basename + '.pdf')
