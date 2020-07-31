from mesa_history import MesaHistory
import matplotlib.pylab as plt
import numpy as np


class MesaHistoryPlot(MesaHistory):
    def times(self):
        izams = np.where(self.center_h1[0] - self.center_h1 > 0.01)[0][0]
        itams = np.where(self.center_h1 < 1.e-4)[0][0]
        izahe = np.where((self.center_he4 < max(self.center_he4) - 1.e-2) &
                      (self.center_h1 < 1.e-6))[0][0]
        itahe = np.where(self.center_he4 < 1.e-4)[0][0]

        return dict(
            izams = izams,
            itams = itams,
            izahe = izahe,
            itahe = itahe,
            )

    def hrd_plot(self, ax = None, color = None, label = None):
        t = self.times()

        if ax is None:
            ax = plt.gca()

        ii = slice(t['izams'], t['itahe'])

        ax.plot(self.log_Teff[ii],
                self.log_L[ii],
                color = color,
                label = label,
                )

        markers = '^vsD'
        times = [t[x] for x in ['izams', 'itams', 'izahe', 'itahe']]
        for m, i in zip(markers, times):
            ax.plot([self.log_Teff[i]],
                    [self.log_L[i]],
                    m,
                    color = color,
                    zorder = 9,
                   )

    def rhot_plot(self, ax = None, color = None, label = None):
        t = self.times()
        if ax is None:
            ax = plt.gca()
        ii = slice(t['izams'], t['itahe'])

        ax.plot(self.log_center_Rho[ii],
                self.log_center_T[ii],
                color = color,
                label = label,
                )

        markers = '^vsD'
        times = [t[x] for x in ['izams', 'itams', 'izahe', 'itahe']]
        for m, i in zip(markers, times):
            ax.plot([self.log_center_Rho[i]],
                    [self.log_center_T[i]],
                    m,
                    color = color,
                    zorder = 9,
                   )
