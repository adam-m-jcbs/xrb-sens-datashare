from matplotlib.font_manager import FontProperties

from ..datainterface import DataInterface

# Later we may want to introduce differnt header types.
class Header():
    def __init__(self,
                 fig = None,  # or require Frame object?
                 **kwargs,
                 ):
        assert fig is not None, 'require figure object'
        self.fig = fig
        self.texts = []

    def clear(self):
        for t in self.texts:
            self.fig.texts.remove(t)
        self.texts = []

    def draw(self):
        self.clear()


class HeaderKep(Header):

    def __init__(self,
                 data = None,
                 **kwargs):

        assert isinstance(data, DataInterface), 'need to provide data source interface DataInterface'
        self.data = data

        super().__init__(**kwargs)

        # this to go to frame?
        # TODO - allow overwrite in rcParam
        self.title_font = FontProperties(
            family = 'monospace',
            style = 'normal',
            stretch = 'normal',
            variant = 'normal',
            weight = 'normal',
            size = 'small',
            )


    def draw(self, *args, **kwargs):
        """
        Draw header.
        """

        self.clear()

        data = self.data
        jm = data.jm
        jmcalc = data.jmcalc
        # the caption
        lines = []
        idt = data.idtcon - 1
        lines.append("{:8s}{:8d}{:23.14e}   {:>3s}({:5d})={:11.4e}     {:24s}".format(
            data.nameprob,
            int(data.ncyc),
            float(data.time),
            data.idtcsym[idt],
            data.jdtc[idt],
            data.dt,
            self.data.datatime))
        lines.append("R ={:11.4e}   Teff ={:11.4e}  L ={:11.4e}   Iter = {:d}".format(
            data.radius,
            data.teff,
            data.xlum,
            data.iter))
        if data.inburn > 0:
            lines[-1] += "   Zb = {:d}   inv = {:d}".format(
                data.nburnz,
                data.ninvl)
        lines.append("Dc ={:11.4e}   Tc ={:11.4e}   Ln ={:11.4e}   Jm ={:5d}".format(
            data.dn[1],
            data.tn[1],
            data.xlumn,
            jm))
        if jm != jmcalc:
            lines[-1] += "({:4d})".format(jmcalc)
        else:
            lines[-1] += "   Etot ={:11.4e}".format(data.ent)

        maxlen = max(len(l) for l in lines)
        for i in range(len(lines)):
            lines[i] += ' ' * (maxlen - len(lines[i]))

        text = self.fig.text(
            0.5, 1., '\n'.join(lines),
            horizontalalignment = 'center',
            verticalalignment = 'top',
            fontproperties = self.title_font,
            )
        self.texts.append(text)
