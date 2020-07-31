"""
Python module to plot KEPLER convection data.

(under construction)
"""

# import physconst
# import os.path

from multiprocessing import Process, Queue, JoinableQueue, cpu_count, active_children

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from operator import xor
from itertools import chain
from collections import deque

from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory, \
     TransformedBbox, Bbox, BboxTransformFrom, BboxTransformTo
from matplotlib.text import Text, TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from matplotlib.legend import Legend

from human import time2human
from utils import cachedmethod, CachedAttribute, queue_processor
from logged import Logged
from physconst import SEC
from physconst import Kepler

from convdata import ConvData, loadconv
from convdata import ConvExtractor, LayerExtractor
from convdata import full_conv_names

from scales import TimeScale

def plotconv(*arg, **kwargs):
    """
    return convplotobject
    """
    if isinstance(arg[0], ConvData):
        c = arg[0]
        arg = arg[1:]
    else:
        c = loadconv(*arg, **kwargs)
    if c is None:
        """
        Error obtaining plot
        """
        return None
    return ConvPlot(c, **kwargs)

plot = plotconv

class Poly(object):
    """
    Growing polygons to which one can add from either side, or join.

    TODO - not add things that could be removed later.
    """
    def __init__(self, outer = True):
        self.outer = outer
        self.x = deque()
        self.y = deque()
    @classmethod
    def start(cls, x, y0, y1, outer = True):
        p = cls(outer)
        p.add_x_2y(x, y0, y1)
        return p
    def add(self, x, y, top = False):
        if top:
            self.x.append(x)
            self.y.append(y)
        else:
            self.x.appendleft(x)
            self.y.appendleft(y)
    def add_x_2y(self, x, y0, y1):
        self.x.appendleft(x )
        self.y.appendleft(y0)
        self.x.append    (x )
        self.y.append    (y1)
    def add_x2(self, x):
        self.y.appendleft(self.y[ 0])
        self.y.append    (self.y[-1])
        self.x.appendleft(x)
        self.x.append    (x)
    def add_x(self, x, top = False):
        if top:
            self.y.append    (self.y[-1])
            self.x.append    (x)
        else:
            self.y.appendleft(self.y[ 0])
            self.x.appendleft(x)
    def __len__(self):
        return len(self.x)

    def simplify1(self):
        """
        Remove duplicates from polygon.

        A place to add the 'reduce' function.
        """
        x = deque()
        y = deque()
        z = zip(self.x,self.y)
        x0, y0 = next(z)
        x.append(x0)
        y.append(y0)
        for x1,y1 in z:
            if (x1 != x0) or (y1 != y0):
                x.append(x1)
                y.append(y1)
                x0,y0 = x1,y1
        self.x = x
        self.y = y

    def simplify2(self,
                 resolution = None):
        """
        Remove points on horz/vert lines.

        A place to add the 'reduce' function.
        """
        x = deque()
        y = deque()
        z = zip(self.x,self.y)
        x0, y0 = next(z)
        x1, y1 = next(z)
        x2, y2 = self.x[-1], self.y[-1]
        x.append(x0)
        y.append(y0)
        for x2,y2 in z:
            if not ((x0 == x1 == x2) or
                    (x0 == x1 == x2)):
                x.append(x1)
                y.append(y1)
                x0,y0 = x1,y1
            x1,y1 = x2,y2
        x.append(x2)
        y.append(y2)
        self.x = x
        self.y = y

    def loy(self):
        return self.y[ 0]
    def hiy(self):
        return self.y[-1]
    def lhy(self, top):
        if top:
            return self.y[-1]
        else:
            return self.y[ 0]
    @classmethod
    def connect(cls, p0,p1,t0,t1):
        """
        connect 2 polygons p0,p1
        at connection points t0,t1
        """
        p = cls(p0.outer)
        p.x = deque(p0.x)
        p.y = deque(p0.y)
        if t0:
            if t1:
                p1.x.reverse()
                p1.y.reverse()
                p.x.extend(p1.x)
                p.y.extend(p1.y)
            else:
                p.x.extend(p1.x)
                p.y.extend(p1.y)
        else:
            if t1:
                assert False, 'This case should not happen'
                p1.x.reverse()
                p1.y.reverse()
                p.x.extendleft(p1.x)
                p.y.extendleft(p1.y)
            else:
                p.x.extendleft(p1.x)
                p.y.extendleft(p1.y)
        # reverse?
        if t0 and t1:
            p.x.reverse()
            p.y.reverse()
            p.outer = p1.outer
        return p

    def path_vertices(self):
        """
        return mpl path vertices for Path including codes
        """
        Pm = Path.MOVETO
        Pl = Path.LINETO
        Pc = Path.CLOSEPOLY
        self.simplify1()
        self.simplify2()
        p = np.ndarray((len(self.x)+1,2), dtype = np.float64)
        p[0:-1,0] = self.x
        p[0:-1,1] = self.y
        p[-1,0]   = self.x[0]
        p[-1,1]   = self.y[0]
        c = [Pm] + [Pl]*(len(self.x)-1) + [Pc]
        if not self.outer:
            p = p[::-1,:]
        return p,c

    def __len__(self):
        return len(self.x)

class Layer(Logged):
    """
    General object for layers.

    need time coordinate - just for poly?
    """
    def __init__(self,
                 ncoord,
                 coord,
                 time = None,
                 label = '',
                 ):
        self.ncoord = ncoord
        self. coord =  coord
        self.  time =   time
        self. label =  label

    # def poly2(self,
    #          time = None,
    #          silent = False):
    #     """
    #     Construct fill polygons.
    #     Holes should have reverse parity.
    #     Return list of polygons.
    #     """
    #     self.setup_logger(silent)
    #     if time is None:
    #         time = self.time
    #     assert time is not None, 'need time'
    #     assert len(time) == len(self.ncoord) + 1, 'need start/stop time for each slice'
    #     # output list
    #     p = list()
    #     # open list
    #     o = list()
    #     for i, nc in enumerate(chain(self.ncoord,[0])):
    #         if len(o) == 0:
    #             # we olny need to start new polygons
    #             for ic in xrange(nc):
    #                 o.append(Poly.start(time[i],*(self.coord[i,ic,:].tolist())))
    #         else:
    #             # let's try from scratch
    #             # we need list of open verices
    #             ol = list()
    #             for v in o:
    #                 v.add_x2(time[i])
    #                 ol.append([v.loy(), v, False])
    #                 ol.append([v.hiy(), v, True ])
    #             ol = sorted(ol,key=lambda v: v[0])
    #             if nc > 0:
    #                 cl = self.coord[i,:,:].reshape(-1)
    #             # DEBUG
    #             else:
    #                 cl = None

    #             # now run through vertices
    #             ic = 0
    #             io = 0
    #             while (ic < 2*nc) or (io < len(ol)):
    #                 if ic < 2*nc-1:
    #                     if (io == len(ol)) or (cl[ic] < ol[io][0]):
    #                         if (io == len(ol)) or (cl[ic + 1] < ol[io][0]):
    #                             # create new poly
    #                             o.append(Poly.start(time[i],cl[ic],cl[ic+1], (ic % 2) == 0))
    #                             ic += 2
    #                             continue
    #                 if io < len(ol)-1:
    #                     if (ic == 2*nc) or (cl[ic] > ol[io + 1][0]):
    #                         if ol[io][1] == ol[io+1][1]:
    #                             # close polygon
    #                             v = ol[io][1]
    #                             o.remove(v)
    #                             p.append(v)
    #                             io += 2
    #                             continue
    #                         # connect polygons
    #                         v0 = ol[io  ][1]
    #                         v1 = ol[io+1][1]
    #                         t0 = ol[io  ][2]
    #                         t1 = ol[io+1][2]
    #                         v = Poly.connect(v0,v1,t0,t1)
    #                         o.remove(v0)
    #                         o.remove(v1)
    #                         o.append(v)
    #                         ol[io  ][1] = None
    #                         ol[io+1][1] = None
    #                         if t0 and t1:
    #                             t = False
    #                         else:
    #                             t = t0
    #                         for ov in ol:
    #                             if ov[1] == v0:
    #                                 ov[1] = v
    #                                 ov[2] = not t
    #                             if ov[1] == v1:
    #                                 ov[1] = v
    #                                 ov[2] = t
    #                         io += 2
    #                         continue
    #                 # add to poly (general case)
    #                 ol[io][1].add(time[i],cl[ic],ol[io][2])
    #                 ic += 1
    #                 io += 1
    #                 continue
    #     self.close_logger(timing='   polygons created in')

    #     # DEBUG
    #     for v in p:
    #         assert(len(v) >= 4), 'Polygon creation failed.'

    #     return p

    def poly(self,
             time = None,
             silent = False):
        """
        Construct fill polygons.
        Holes should have reverse parity.
        Return list of polygons.
        """
        self.setup_logger(silent)
        if time is None:
            time = self.time
        assert time is not None, 'need time'
        assert len(time) == len(self.ncoord) + 1, 'need start/stop time for each slice'
        # output list
        p = list()
        # open list
        ol = list()
        for i, nc in enumerate(chain(self.ncoord,[0])):
            t = time[i]
            if len(ol) == 0:
                # we only need to start new polygons
                for ic in range(nc):
                    v = Poly.start(t,*(self.coord[i,ic,:].tolist()))
                    ol.extend([[v.loy(), v, False],
                               [v.hiy(), v, True ]])
            else:
                # we need list of open verices
                for vl in ol:
                    vl[1].add_x(t,vl[2])
                    vl[0] = vl[1].lhy(vl[2])
                if nc > 0:
                    cl = self.coord[i,:,:].reshape(-1)
                # DEBUG
                else:
                    cl = None

                # now run through vertices
                ic = 0
                io = 0
                while (ic < 2*nc) or (io < len(ol)):
                    if ic < 2*nc-1:
                        if (io == len(ol)) or (cl[ic] < ol[io][0]):
                            if (io == len(ol)) or (cl[ic + 1] < ol[io][0]):
                                # create new poly
                                v = Poly.start(t,cl[ic],cl[ic+1], (ic % 2) == 0)
                                ol[io:io] = [[v.loy(), v, False],
                                             [v.hiy(), v, True ]]
                                io += 2
                                ic += 2
                                continue
                    if io < len(ol)-1:
                        if (ic == 2*nc) or (cl[ic] > ol[io + 1][0]):
                            if ol[io][1] == ol[io+1][1]:
                                # close polygon
                                v = ol[io][1]
                                p.append(v)
                                del ol[io:io+2]
                                continue
                            # connect polygons
                            v0, t0 = ol[io  ][1:3]
                            v1, t1 = ol[io+1][1:3]
                            v = Poly.connect(v0,v1,t0,t1)
                            del ol[io:io+2]
                            if t0 and t1:
                                tv = False
                            else:
                                tv = t0
                            for ov in ol:
                                if ov[1] == v0:
                                    ov[1:3] = [v, not tv]
                                if ov[1] == v1:
                                    ov[1:3] = [v, tv]
                            continue
                    # add to poly (general case)
                    ol[io][1].add(t,cl[ic],ol[io][2])
                    ic += 1
                    io += 1

                    continue
        self.close_logger(timing='polygons {:<9s} created   in'.format(self.label))

        # DEBUG
        for v in p:
            assert(len(v) >= 4), 'Polygon creation failed.'
        assert len(ol) == 0, 'Not all polygons were closed.'

        return p

    def poly_path(self,
                  time = None,
                  silent = False):
        """
        Return polygon path for layer.
        """
        poly = self.poly(time,
                         silent = silent)
        self.setup_logger(silent)
        p = np.ndarray((0,2), dtype = np.float64)
        c = list()
        for v in poly:
            px, cx = v.path_vertices()
            # this may be slow:
            # NOTE: this _IS_ slow
            c += cx
            p = np.append(p, px, axis=0)
        # does the following still work:
        c = np.array(c)

        path = Path(p, codes=c)
        self.close_logger(timing='path     {:<9s} created   in'.format(self.label))
        return path

# def ltest(num = 0):
#     if num == 0:
#         ncoord = np.array([2,2,2,0,0,0,0,0,0,0])
#         coord = np.zeros((10,10,2))
#         coord[0,0,:] = [0.5,2.]
#         coord[0,1,:] = [2.25,2.75]
#         coord[1,0,:] = [0.5,1.25]
#         coord[1,1,:] = [1.75,2.5]
#         coord[2,0,:] = [1.,2.]
#         coord[2,1,:] = [2.25,3.5]
#         time = [0,1,2,3,4,5,6,7,8,9,10]
#         layer = Layer(ncoord,coord).poly_path(time)
#         patch = PathPatch(layer)
#         plt.clf()
#         ax = plt.gca()
#         ax.add_patch(patch)
#         ax.set_xlim(-1,4)
#         ax.set_ylim(0,4)
#         plt.draw()
#         return Layer(ncoord,coord).poly(time)


# We want to define some transformation object
# with unit
# ... or just a name for axis


class Scale(object):
    def __init__(self):
        pass

class LayerProcessor(Logged):
    """
    Process layers and provide logging interface / output.
    """
    def __init__(self,
                 data):
        self.data = data
        self.time = data.get('time', None)
    def __call__(self, task):
        task = task.copy()
        task.update(self.data)
        silent = task.get('silent',   None)
        self.setup_logger(silent)
        level   = task.get('level',   None)
        ilayer  = task.get('ilayer',  None)
        olayer  = task.get('olayer',  None)
        color   = task.get('color',   None)
        label   = task.get('label',   None)
        extractor = task.get('extractor',   None)
        self.logger.info('Processing layer {:s}.'.format(label))
        layer = globals()[extractor](**task).layer_data
        path = Layer(*layer, label = label).poly_path(self.time)
        self.close_logger(timing = '{:s} processed in'.format(label))
        return (path, color, olayer)

class ConvPlot(Logged):
    """
    A first test for a plot.
    """
    conv_hatch = {
        'conv' : dict(color = 'green',
                      hatch  = '/',
                      fill   = False,
                      zorder = 2),
        'semi' : dict(color  = 'red',
                      hatch  = 'XX',
                      lw     = 0,
                      fill  = False,
                      zorder = 2),
        'neut' : dict(color  = 'cyan',
                      hatch  = 'X',
                      lw     = 0,
                      fill   = False,
                      zorder = 2),
        'osht' : dict(color  = 'magenta',
                      hatch  = '++',
                      lw     = 0,
                      fill   = False,
                      zorder = 2),
        'thal' : dict(color  = 'yellow',
                      hatch  = '\\',
                      lw     = 0,
                      fill   = False,
                      zorder = 2)}

    gain_layer = dict(facecolor = 'blue',
                      edgecolor = 'none',
                      lw        = 0,
                      alpha     = 0.1,
                      zorder    = 1)

    loss_layer = dict(facecolor = 'magenta',
                      edgecolor = 'none',
                      lw        = 0,
                      alpha     = 0.1,
                      zorder    = 1)

    subs_layer = dict(facecolor = 'black',
                      edgecolor = 'none',
                      lw        = 0,
                      alpha     = 0.2,
                      zorder    = 1)


    def __init__(self,
                 convdata = None,
                 quant = 'nuc',
                 # mass = 'sun',  # this should become the interface
                 # time = 'year',
                 radius = None,
                 silent = False,
                 mingain = None,
                 minloss = None,
                 maxgain = None,
                 maxloss = None,
                 reversetime = False,
                 showmodels = False,
                 showlegend = True,
                 showconvlegend = True,
                 showcore = False,
                 decmass = False,
                 logtime = None,
                 logtimeunit = None,
                 timecc = False,
                 xlim = None,
                 ylim = None,
                 surface = False,
                 logarithmic = False,
                 column = False,
                 stability = None,
                 solar = False,
                 processes = -1.5, # negative values mean * # virtual cores
                 ymax = None,
                 ymin = None,
                 figsize = (8,6),
                 figure = None,
                 axes = None,
                 dpi = 102,
                 full_conv_names = False,
                 interactive = True,
                 auto_position = True,
                 ):
        """
        We now have mingain, minloss in log energy units

        Currently non-integer levels are not supported, neither
        for the data not for the paramenter.

        Need to rewrite coordinate system settings to more options and
        to be resonable.
        """

        self.setup_logger(silent)

        if isinstance(convdata, str):
            convdata = ConvData(convdata)
        if convdata is None and 'convdata' in self.__dict__:
            convdata = self.convdata
        self.convdata = convdata
        self.showmodels = showmodels
        self.showlegend = showlegend
        self.showconvlegend = showconvlegend
        self.showcore = showcore
        self.full_conv_names = full_conv_names

        if figure is None:
            if axes is not None:
                self.fig = axes.figure
            else:
                self.fig = plt.figure(
                    figsize = figsize,
                    dpi = dpi,
                    facecolor = 'white',
                    edgecolor = 'white'
                    )
        else:
            self.fig = figure

        if axes is None:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = axes
        self.auto_position = auto_position
        self._set_position()

        layer_index, layer_radial, layer_name = convdata.layer_index(quant)
        loss_max, gain_max = convdata.level_range(quant)
        layer_gain_min, layer_loss_min = convdata.level_min(quant)

        self.quant = quant
        self.layer_radial = layer_radial
        if radius is None:
            radius = layer_radial
        self.radius = radius

        mscale = None
        ycoord = 'mass'
        if radius:
            ycoord = 'radius'
        if column:
            ycoord = 'column'
        if solar:
            ycoord = 'msun'
            mscale = 1 / Kepler.solmass

        if mingain is None:
            mingain = layer_gain_min
        if minloss is None:
            minloss = layer_loss_min

        if maxgain is None:
            maxgain = layer_gain_min + gain_max - 1
        if maxloss is None:
            maxloss = layer_loss_min + loss_max - 1

        minloss = max(minloss, layer_gain_min)
        mingain = max(mingain, layer_gain_min)

        self.mingain = mingain
        self.minloss = minloss
        self.maxgain = maxgain
        self.maxloss = maxloss

        min_gain_level = mingain - layer_gain_min + 1
        max_gain_level = maxgain - layer_loss_min + 1
        min_loss_level = minloss - layer_gain_min + 1
        max_loss_level = maxloss - layer_loss_min + 1

        if stability is None:
            stability = list(self.conv_hatch.keys())
        conv_max = len(self.conv_hatch)

        self.stability = stability

        # deal with time axis
        xscale = None
        xlabel = None

        if logtime == 'models':
            xscale = 'unitscale'
            xscale_args = dict(
                var = 'model number',
                unit = None,
                )
            xtime = convdata.xmodels
            # xlabel = 'model number'
        elif logtime is not None:
            if logtimeunit is None:
                logtimeunit = 'yr'
            if logtimeunit == 'yr':
                timeunit = SEC
            elif logtimeunit == 'd':
                timeunit = 86400
            elif logtimeunit == 'h':
                timeunit = 3600
            elif logtimeunit == 'min':
                timeunit = 60
            elif logtimeunit == 's':
                timeunit = 1
            else:
                raise NotImplementedError('invalid logtimeunit')
            # to be compatible with IDL/plotconv we use log(time/yr)
            xtime = np.log10(convdata.xtimecc(10**logtime * SEC)) - np.log10(timeunit)
            xlabel = r'log( time till core collapse / {} )'.format(logtimeunit)
        elif not timecc in (None, False):
            xtime = convdata.xtimecc()

            # if timecc == 'days':
            #     xtimeunit = 'days'
            #     xtimefac = 1/86400
            # else:
            #     xtimeunit = 's'
            #     xtimefac = 1
            # xtime *= xtimefac
            # xlabel = r'time till core collapse ({})'.format(xtimeunit)

            xscale = 'timescale'
            xscale_args = dict(var = 'time till core collapse')
            # xscale_args = dict(var = 'time BC')
        else:
            xscale = 'timescale'
            xscale_args = dict(var = 'time')
            xtime = convdata.xtime

        xrange = [xtime[0], xtime[-1]]
        self.xtime = xtime

        if reversetime:
            xrange = xrange[::-1]

        if xlim is not None:
            xrange = xlim

        self.layers = dict(
            gain = np.tile(None, gain_max),
            loss = np.tile(None, loss_max),
            conv = np.tile(None, conv_max),
            )

        offset = None
        if decmass:
            if ycoord == 'mass':
                offset = -convdata.dmdec
            elif ycoord == 'column':
                offset = -convdata.dcdec

        if processes is None:
            processes = cpu_count()
        if processes < 0:
            processes = -processes * cpu_count()
        maxproc = (len(stability) +
                   max_gain_level - min_gain_level + 1 +
                   max_loss_level - min_loss_level + 1)
        nproc = min(int(processes), maxproc)

        self.add_timer('process')
        self.logger.info('Setting up {:d} worker processes.'.format(nproc))
        task_queue = JoinableQueue()
        done_queue = Queue()
        args = dict(
            ydata = convdata.data,
            radius  = radius,
            column  = column,
            surface = surface,
            offset = offset,
            logarithmic = logarithmic,
            scale = mscale,
            time = xtime,
            silent = silent,
            )
        params = dict(
            processor = LayerProcessor,
            data = args,
            )
        nin = 0
        for i in range(nproc):
            p = Process(
                target = queue_processor,
                args = (
                    task_queue,
                    done_queue,
                    params,
                    ))
            p.start()
        self.close_timer('process', 'Worker processes set up in {:s}')
        self.add_timer('setup')
        self.logger.info('Setting up tasks.')

        for conv in stability:
            iconv, conv_sentinel, conv_name = ConvData.conv_index(conv)
            task = dict(
                ilayer = iconv,
                olayer = ('conv', iconv-1),
                color  = self.conv_hatch[conv],
                label  = 'conv  {:>3d}'.format(iconv),
                extractor = 'ConvExtractor',
                )
            task_queue.put(task)
            nin += 1

        for level in range(min_gain_level, max_gain_level+1):
            task = dict(
                ilayer = layer_index,
                olayer = ('gain', +level-1),
                color  = self.gain_layer,
                level  = level,
                label = 'level {:>3d}'.format(level),
                extractor = 'LayerExtractor',
                )
            task_queue.put(task)
            nin += 1

        for level in range(-min_loss_level, -max_loss_level-1, -1):
            task = dict(
                ilayer = layer_index,
                color  = self.loss_layer,
                olayer = ('loss', -level-1),
                level  = level,
                label = 'level {:>3d}'.format(level),
                extractor = 'LayerExtractor',
                )
            task_queue.put(task)
            nin += 1

        self.logger.info('tasks set up in {:s}'.format(time2human(self.finish_timer('setup'))))
        self.logger.info('Running tasks.')
        self.add_timer('task')

        task_queue.join()

        self.logger.info('tasks finished in {:s}'.format(time2human(self.finish_timer('task'))))
        self.add_timer('layer')
        self.logger.info('Plotting layers.')

        vtot = 0
        for i in range(nin):
            l  = done_queue.get()
            # this is illegible ... replace "l" by object/dict
            vtot += len(l[0])
            patch = PathPatch(l[0],**l[1])
            self.ax.add_patch(patch)
            self.layers[l[2][0]][l[2][1]] = patch
            nin -= 1
        assert nin == 0

        self.logger.info('{:d} vertices plotted'.format(vtot))

        for i in range(nproc):
            task_queue.put('STOP')

        task_queue.join()
        task_queue.close()
        done_queue.close()
        self.logger.info('Plotting layers finished in {:s}'.format(time2human(self.finish_timer('layer'))))

        # TODO
        # all of these need use proper "scale" classes
        # need to add column depth and log col depth (log mass ?)

        # line for total mass/radius
        yrange = None
        yscale = None
        ylabel = None
        if ycoord == 'msun':
            ylabel = r'enclosed mass (solar masses)'
            ystar = np.insert(convdata.xmstar, 0, convdata.xmstar[0]) * mscale
        elif ycoord == 'mass':
            ylabel = r'enclosed mass (g)'
            ystar = np.insert(convdata.xmstar,0,convdata.xmstar[0])
            if decmass:
                ystar -= convdata.dmdec
        elif ycoord == 'radius':
            ylabel = r'radius coordinate (cm)'
            ystar = np.insert(convdata.rstar,0,convdata.rstar[0])
        elif ycoord == 'column':
            # ylabel = r'column thickness (g cm$^{-2}$)'
            ystar = np.insert(convdata.cstar, 0, convdata.cstar[0])
            if decmass:
                ystar -= convdata.dcdec
            yscale = 'unitscale'
            yscale_args = dict(
                var = 'column thickness',
                unit = r'g cm$^{-2}$',
                )
        else:
            raise Exception('Unknown coordinate type "{:}".'.format(ycoord))
        if not yrange:
            yrange = np.array([0., np.max(ystar)])

        if ymin is not None:
            yrange[0] = ymin
        if ymax is not None:
            yrange[1] = ymax

        if surface:
            yrange = yrange[::-1]
            ystar = np.zeros_like(ystar)
            if ycoord == 'msun':
                ylabel = r'mass belowe surface (solar masses)'
            if ycoord == 'mass':
                ylabel = r'mass belowe surface (g)'
            elif ycoord == 'radius':
                ylabel = r'radius below surface (cm)'
            elif ycoord == 'column':
                ylabel = r'column depth (g cm$^{-2}$)'

        logmin = 1.e-99

        if logarithmic:
            ystar = np.log10(np.maximum(ystar, logmin))
            ylabel = r'log ' + ylabel
            yrange = np.log10(np.maximum(yrange, logmin))

        if ylim is not None:
            yrange = ylim

        self.ystar = ystar

        self.show_surf(xtime, ystar)

        self.ax.set_xlim(*xrange)
        self.ax.set_ylim(*yrange)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if xscale:
            self.ax.set_xscale(xscale, **xscale_args)
        if yscale:
            self.ax.set_yscale(yscale, **yscale_args)

        self.show_legend(showlegend, draw = False)
        # if showlegend:
        #     self.legend = self.LevelLegend(self)
        # else:
        #     self.legend = None

        if showconvlegend:
            self.convlegend = self.ConvLegend(self)
        else:
            self.convlegend = None

        if showmodels:
            self.models = self.ModelsLegend(self)
        else:
            self.models = None

        if showcore:
            self.core = self.Core(self)
        else:
            self.core = None

        for p in active_children():
            self.logger.info('terminating process {:d}'.format(p.pid))
            p.terminate()
        self.close_logger(timing = 'convplot   constructed in')

        if interactive:
            self.fig.canvas.mpl_connect('resize_event', self._on_resize)

        plt.draw()


    def show_surf(self,
                  xtime,
                  ystar,
                  smooth = False,
                  color = 'black',
                  linewidth = 2):
        """
        show surface of star
        """
        if not smooth:
            i = np.arange(xtime.size)
            ii = np.vstack([i,i]).transpose().flatten()
            xtime = xtime[ii[:-1]]
            ystar = ystar[ii[1:]]
            # xtime = np.vstack([xtime,xtime]).transpose().flatten()[:-1]
            # ystar = np.vstack([ystar,ystar]).transpose().flatten()[1:]
        line = Line2D(xtime,
                      ystar,
                      color = color,
                      linewidth = linewidth)
        self.ax.add_line(line)



    def _set_position(self):
        if not self.auto_position:
            return
        self.box = np.array([0,0,1,1], dtype = np.float64)
        heights = self._fonts_heights()
        h = heights['fh']
        v = heights['fv']
        self.box[0] += 6.3 * v
        self.box[1] += 4.9 * h
        if self.showlegend:
            self.box[2] -= 3.5 * v
        else:
            self.box[2] -= 0.15 * v
        if self.showmodels:
            self.box[3] -= 3 * h
        else:
            self.box[3] -= 1.7 * h
        self.box[2:4] -= self.box[0:2]
        self.ax.set_position(self.box)

    def show_models(self,
                    show = None,
                    draw = True):
        """
        show/hide models
        """
        if show is None:
            show = not self.showmodels
        if show:
            if self.models is None:
                self.models = self.ModelsLegend(self)
        self.showmodels = show
        if self.models is not None:
            self.models.update()
        self._set_position()
        if draw:
            self.ax.figure.canvas.draw()

    def show_convlegend(self,
                    show = None,
                    draw = True):
        """
        show/hide convection legend
        """
        if show is None:
            show = not self.showconvlegend
        if show:
            if self.convlegend is None:
                self.convlegend = self.ConvLegend(self)
        self.showconvlegend = show
        if self.convlegend is not None:
            self.convlegend.update_visible()
        if draw:
            self.ax.figure.canvas.draw()

    def show_legend(self,
                    show = None,
                    draw = True):
        """
        show/hide level legend
        """
        if 'legend' not in self.__dict__:
            self.legend = None
        if show is None:
            show = not self.showlegend
        if show:
            if self.legend is None:
                self.legend = self.LevelLegend(self)
                if self.legend.empty:
                    self.legend.clear()
                    self.legend = None
                    show = False
        else:
            if self.legend is not None:
                self.legend.clear()
                self.legend = None
        self.showlegend = show
        self._set_position()
        if draw:
            self.ax.figure.canvas.draw()

    def update_stability(self,
                         stab,
                         value):
        if value is not None:
            if value:
                if stab not in self.stability:
                    self.stability.append(stab)
            else:
                if stab in self.stability:
                    self.stability.remove(stab)

    def update(self,
               maxgain = None,
               maxloss = None,
               mingain = None,
               minloss = None,
               stability = None,
               showmodels = None,
               showconvlegend = None,
               showlegend = None,
               showcore = None,
               conv = None,
               thal = None,
               semi = None,
               neut = None,
               osht = None,
               trange = None,
               yrange = None,
               draw = True):

        if maxgain is not None:
            self.maxgain = maxgain
        if maxloss is not None:
            self.maxloss = maxloss
        if mingain is not None:
            self.mingain = mingain
        if minloss is not None:
            self.minloss = minloss
        if stability is not None:
            self.stability = stability

        self.update_stability('conv', conv)
        self.update_stability('thal', thal)
        self.update_stability('semi', semi)
        self.update_stability('osht', osht)
        self.update_stability('neut', neut)

        if showmodels is not None:
            self.show_models(
                showmodels,
                draw = False)
        if showlegend is not None:
            self.show_legend(
                showlegend,
                draw = False)
        if showconvlegend is not None:
            self.show_convlegend(
                showconvlegend,
                draw = False)
        if showcore is not None:
            self.show_core(
                showcore,
                draw = False)

        self.update_levels()

        if trange is not None:
            self.ax.set_xlim(*trange)
        if yrange is not None:
            self.ax.set_ylim(*yrange)

        if draw:
            self.ax.figure.canvas.draw()


    def update_levels(self):
        """
        Update visible levels from [min/max][gain/loss] and stability.
        """
        quant = self.quant
        convdata = self.convdata

        layer_index, layer_radial, layer_name = self.convdata.layer_index(quant)
        loss_max, gain_max = convdata.level_range(quant)
        layer_gain_min, layer_loss_min = convdata.level_min(quant)

        for conv in self.conv_hatch.keys():
            iconv, conv_sentinel, conv_name = ConvData.conv_index(conv)
            patch = self.layers['conv'][iconv-1]
            if patch is not None:
                patch.set_visible(conv_name in self.stability)
        for level in range(gain_max):
            patch = self.layers['gain'][level]
            assert patch is not None,'patch not loaded.'
            patch.set_visible(self.mingain <= level + layer_gain_min <= self.maxgain)
        for level in range(loss_max):
            patch = self.layers['loss'][level]
            assert patch is not None,'patch not loaded.'
            patch.set_visible(self.minloss <= level + layer_loss_min <= self.maxloss)

        if self.legend is not None:
            self.legend.redraw()
        if self.convlegend is not None:
            self.convlegend.update()

    class ConvLegend(Logged):
        """
        Class to display convection legend.
        """
        def __init__(self, parent):
            self.parent = parent
            self.draw()
            self.update_visible()
        def update(self):
            self.parent.ax.artists.remove(self.legend)
            self.draw()
            self.update_visible()
        def show():
            self.parent.showconvlegend = True
            self.update_visible(self)
        def hide():
            self.parent.showconvlegend = False
            self.update_visible(self)
        def update_visible(self):
            show = self.parent.showconvlegend
            self.legend.set_visible(show)
        def draw(self):
            patches = list()
            labels  = list()
            parent = self.parent
            for conv in parent.conv_hatch.keys():
                iconv, conv_sentinel, conv_name = ConvData.conv_index(conv)
                patch = parent.layers['conv'][iconv-1]
                if patch is not None:
                    if patch.get_visible():
                        patches.append(parent.layers['conv'][iconv-1])
                        if parent.full_conv_names:
                            conv_name = full_conv_names[conv_sentinel]
                        labels.append(conv_name)
            legend = Legend(
                parent.ax, patches, labels,
                loc = 'best',
                )
            legend.set_draggable(True)
            parent.ax.add_artist(legend)
            self.legend = legend

    class ModelsLegend(Logged):
        def update_trans(self, ax):
            # trans_top_px = (ax.figure.transFigure.inverted() +
            #                 Affine2D.from_values(1,0,0,-1,0,1) +
            #                 ax.figure.transFigure)
            # self.trans_top_px.set(trans_top_px)
            pass

        def clear(self):
            """
            clear model patches and texts
            """
            if '_elements' not in self.__dict__:
                return
            ax = self.parent.ax
            for item in self._elements:
                if isinstance(item, Patch):
                    ax.patches.remove(item)
                elif isinstance(item, Text):
                    ax.artists.remove(item)
                else:
                    raise Exception('Unknown Legend Element')
            del self._elements

        def update(self):
            show = self.parent.showmodels
            for item in self._elements:
                item.set_visible(show)
        def __init__(self,
                     parent,
                     silent = False):
            self.setup_logger(silent)
            self.logger.info('Plotting models')

            self._elements = list()
            self.parent = parent
            convdata = parent.convdata
            xtime = parent.xtime

            heights = parent._fonts_heights()
            h = heights['h']

            ax = parent.ax
            figbox = ax.figbox
            base = 0
            height = 2.5*h
            ytick0 = base + height
            colors = ['black','red','green','blue']

            # trans_top_px = (ax.figure.transFigure.inverted() +
            #                 Affine2D.from_values(1,0,0,-1,0,1) +
            #                 ax.figure.transFigure).frozen()
            trans_top_px = (BboxTransformFrom(ax.figure.bbox) +
                            Affine2D.from_values(1,0,0,-1,0,1) +
                            BboxTransformTo(ax.figure.bbox))
            self.trans_top_px = trans_top_px

            trans = blended_transform_factory(
                ax.transData,
                trans_top_px)
            bbox = Bbox([[0,ytick0],[1,0]])
            trans_text = blended_transform_factory(
                ax.transAxes,
                ax.figure.transFigure)
            trans_cbox = blended_transform_factory(
                ax.transAxes,
                trans_top_px)
            cbox = TransformedBbox(bbox,
                                   trans_cbox)

            maxlev = int(np.log10(convdata.data[-1].ncyc)*2)+1
            nmodels = convdata.nmodels
            nlines = np.zeros(maxlev, dtype=np.int)
            times = np.ndarray((maxlev,nmodels))

            Pm = Path.MOVETO
            Pl = Path.LINETO
            seg = np.array([Pm,Pl])

            for time, data in zip(xtime[1:],convdata.data):
                mag = 0
                val = data.ncyc
                while (val % 10) == 0:
                    mag += 1
                    val //= 10
                half = int((val % 5) == 0)
                lev = 2 * mag + half
                times[lev,nlines[lev]] = time
                nlines[lev] += 1
            codes = np.ndarray((maxlev,nmodels*2))
            paths = np.ndarray((maxlev,nmodels*2,2))
            for lev in range(maxlev):
                if nlines[lev] > 0:
                    codes = np.tile(seg,nlines[lev])
                    paths = np.ndarray((nlines[lev]*2,2))
                    paths[0::2,0] = times[lev,0:nlines[lev]]
                    paths[1::2,0] = times[lev,0:nlines[lev]]
                    # paths[:,0] = np.tile(times[lev,0:nlines[lev]],2).reshape(2,-1).transpose().reshape(-1)
                    paths[0::2,1] = ytick0
                    mag  = (lev // 2)
                    half = (lev % 2)
                    paths[1::2,1] = ytick0 - (3 + 1.5*mag + half) * 0.05 * height
                    path = Path(paths,
                                codes = codes)
                    patch = PathPatch(path,
                                      linewidth = 0.1 * (1 + mag + half),
                                      color = colors[mag % len(colors)],
                                      transform = trans)
                    ax.add_patch(patch)
                    patch.set_clip_box(cbox)
                    self._elements.append(patch)

            fp = FontProperties()
            fp.set_size(0.71 * fp.get_size())
            text = Text(x = 0.5,
                        y = 1,
                        text = 'models',
                        color = 'black',
                        clip_on = False,
                        ha = 'center',
                        va = 'top',
                        fontproperties = fp,
                        transform = trans_text)
            ax.add_artist(text)
            self._elements.append(text)
            self.close_logger(timing = 'Plotting models finished in')

    def _on_resize(self, event):
         # print('resize', event.__dict__)
         self._set_position()
         if self.legend is not None:
             self.legend.redraw()
         if self.models is not None:
             self.models.update_trans(self.ax)

    def _fonts_heights(self,
                       fp = None):
        """
        Somehow we need to know font sizes in axes and figure (and
        data?) coordinates.

        Obviouslty, axis coordinates are only useful after figure
        coordinates (set_position) have been established, the
        values will change if the axis size (in device
        coordinates) chages;

        Figure coordinates will have to be re-computed when the figure
        size changes.  data coordinates will need to be updated
        essentially when anything changes.

        """
        if fp is None:
            fp = FontProperties()
        tp = TextPath(xy=(0,0),s = 'Xg',fontproperties = fp)
        # this gives coordinates in device units. so we translate those back
        h = np.array([tp.vertices.min(axis=0),
                      tp.vertices.max(axis=0)])

        # first the non-rotated horizontal ones
        # data
        dh = self.ax.transData.inverted().transform(h)
        dh_height = dh[1,1]-dh[0,1]
        # axes
        ah = self.ax.transAxes.inverted().transform(h)
        ah_height = ah[1,1]-ah[0,1]
        # figure
        fh = self.fig.transFigure.inverted().transform(h)
        fh_height = fh[1,1]-fh[0,1]
        # pixel (device) height
        h_height = h[1,1]-h[0,1]

        # now rotated Vertical data
        v = h[:,::-1]
        # data
        dv = self.ax.transData.inverted().transform(v)
        dv_height = dv[1,0]-dv[0,0]
        # axes
        av = self.ax.transAxes.inverted().transform(v)
        av_height = av[1,0]-av[0,0]
        # figure
        fv = self.fig.transFigure.inverted().transform(v)
        fv_height = fv[1,0]-fv[0,0]
        # pixel (device) height
        v_height = h[1,0]-h[0,0]

        return dict(
            h  =  h_height,
            dh = dh_height,
            ah = ah_height,
            fh = fh_height,
            v  =  v_height,
            dv = dv_height,
            av = av_height,
            fv = fv_height)


    class LevelLegend(object):
        """
        Class do plot legend.
        """

        def _text(self,
                  s = '',
                  x = None,
                  y = None,
                  dx = None,
                  dy = None,
                  pathfont = True,
                  ha = 'center',
                  va = 'center',
                  fp = None,
                  border = 0.1,
                  color = 'black',
                  use_scale = False,
                  scale_only = False):
            """
            for now we do only vertical text
            """
            ax = self.parent.ax
            if pathfont:
                text = TextPath(xy=(0,0),
                                s = s,
                                fontproperties = fp)
                fxmin, fymin = text.vertices.min(axis=0)
                fxmax, fymax = text.vertices.max(axis=0)
                fwidth = fxmax - fxmin
                fheight = fymax - fymin

                if use_scale:
                    scale = self._legend_font_scale
                else:
                    tax = ax.transAxes.inverted().transform([[0,0],[1,1]])
                    sax = (tax[1,:]-tax[0,:]).reshape(-1)
                    cax = np.array([fheight,fwidth])*sax
                    box = np.array([dx, dy])*(1-2*border)
                    xscale = box/cax
                    fscale = np.min(xscale)
                    scale = fscale*sax[::-1]
                    self._legend_font_scale = scale
                    self.fscale = fscale

                if scale_only:
                    return

                if ha == 'center':
                    y_offset = -fymin + 0.5 * -fheight
                elif ha == 'left':
                    y_offset = -fymax
                elif ha == 'right':
                    y_offset = -fymin
                else:
                    raise Exception('undefined')
                if va == 'center':
                    x_offset = -fxmin + 0.5 * -fwidth
                elif va == 'bottom':
                    x_offset = -fxmin
                elif va == 'top':
                    x_offset = -fxmax
                else:
                    raise Exception('undefined')

                path = text.transformed(
                    Affine2D()
                    .translate(x_offset, y_offset)
                    .scale(*scale)
                    .rotate_deg(90.)
                    .translate(x, y))
                patch = PathPatch(
                    path,
                    clip_on = False,
                    facecolor=color,
                    edgecolor='none',
                    lw = 0,
                    alpha=1,
                    transform = ax.transAxes)
                ax.add_patch(patch)
                self._elements.append(patch)
            else:
                text = Text(
                    x = x,
                    y = y,
                    text = s,
                    color = color,
                    clip_on = False,
                    ha = ha,
                    va = va,
                    rotation = 'vertical',
                    fontproperties = fp,
                    transform = ax.transAxes)
                ax.add_artist(text)
                self._elements.append(text)

        def redraw(self):
            self.clear()
            self.draw()

        def clear(self):
            """
            clear bar
            """
            if '_elements' not in self.__dict__:
                return
            ax = self.parent.ax
            for item in self._elements:
                if isinstance(item, Patch):
                    ax.patches.remove(item)
                elif isinstance(item, Text):
                    ax.artisis.remove(item)
                else:
                    raise Exception('Unknown Legend Element')
            del self._elements

        def __init__(self,
                     parent  = None,
                     fontsize = 0):
            """
            Initialize energy level bar.
            """
            # TODO
            # need to react to draw/resize
            # maybe deal with "select" levels?
            # make same font size for all annotations

            self.parent   = parent
            self.fontsize = fontsize
            self.empty    = False

            self.draw()

        def draw(self):
            self.clear()
            self._elements = list()

            ax       = self.parent.ax
            mingain  = self.parent.mingain
            minloss  = self.parent.minloss
            maxgain  = self.parent.maxgain
            maxloss  = self.parent.maxloss
            radius   = self.parent.layer_radial

            fontsize = self.fontsize

            figbox = ax.figbox

            pathfont = (fontsize == 0)
            if fontsize <= 0:
                fp = FontProperties()
            else:
                fp = FontProperties(size = fontsize)

            base = 1
            width = (1 - figbox.x1) / figbox.width
            border = 0.1*width
            # alternatively, we could make a 1px border
            # db = ax.transAxes.inverted().transform([[0,0],[1,1]])
            # border = 5*(db[1,0]-db[0,0])
            # does not work because transaxis changes when box is set

            thick = (width - 3*border) / 2
            fborder = 0.1


            x0 = base + border
            xf = base + thick + 2*border

            ngain =  maxgain - mingain + 1
            nloss =  maxloss - minloss + 1
            nlevel = max(0,ngain) + max(nloss,0)

            showzero = (ngain > 0) and (nloss > 0)
            nlevel += int(showzero)

            if nlevel <= 0:
                self.empty = True
                return

            dy = 1/nlevel

            fcolors = ['black','white']
            fclim = 5

            text_font_kw = dict(
                x = x0 + 0.5*thick,
                dx = thick,
                dy = dy,
                ha = 'center',
                va = 'center',
                use_scale = False,
                fp = fp,
                border = fborder,
                pathfont = pathfont)

            # we want to find scaling
            lmin = 0
            lmax = 0
            if ngain > 0:
                lmin = mingain
                lmax = maxgain
            if nloss > 0:
                if lmin is None:
                    lmin = minloss
                    lmax = maxloss
                else:
                    lmin = min(lmin,minloss)
                    lmax = max(lmax,maxloss)
            ltemp = '9'
            if max(abs(lmin),abs(lmax)) > 9:
                ltemp = '99'
            if lmin < 0:
                if ltemp == '9':
                    ltemp = '-9'
                if ltemp == '99' and lmin < -9:
                    ltemp = '-99'
            self._text(ltemp,
                       scale_only = True,
                       **text_font_kw)
            text_font_kw['use_scale'] = True

            if self.fscale < 1:
                thick *= self.fscale
                width *= self.fscale
                border *= self.fscale
                x0 = base + border
                xf = base + thick + 2*border
                text_font_kw['x'] = x0 + 0.5*thick
                self.parent.box = [figbox.x0,
                                   figbox.y0,
                                   1-(1-figbox.x1)*self.fscale-figbox.x0,
                                   figbox.height]
                ax.set_position(self.parent.box)
            for i in range(ngain):
                rect = Rectangle((x0,1),
                                 thick,
                                 (i - ngain)*dy,
                                 clip_on = False,
                                 transform = ax.transAxes,
                                 **self.parent.gain_layer)
                ax.add_patch(rect)
                self._elements.append(rect)

                self._text(
                    s = str(mingain + i),
                    y = 1 + (i - ngain + 0.5)*dy,
                    color = fcolors[i > fclim],
                    **text_font_kw)

            for i in range(nloss):
                rect = Rectangle((x0,0),
                                 thick,(nloss-i)*dy,
                                 clip_on = False,
                                 transform = ax.transAxes,
                                 **self.parent.loss_layer)
                ax.add_patch(rect)
                self._elements.append(rect)

                self._text(
                    s = str(minloss + i),
                    y = (nloss - i - 0.5)*dy,
                    color = fcolors[i > fclim],
                    **text_font_kw)

            text_font_kw['color'] = 'black'
            if showzero:
                self._text(
                    s = '...',
                    y = (max(0,nloss) + 0.5)*dy,
                    **text_font_kw)

            text_font_kw['x'] = xf
            text_font_kw['ha'] = 'left'
            del text_font_kw['va']
            if (nloss > 0) and (nloss > 0):
                self._text(
                    s = 'LOSS',
                    y = 0,
                    va = 'bottom',
                    **text_font_kw)
                self._text(
                    s = 'GAIN',
                    y = 1,
                    va = 'top',
                    **text_font_kw)
            legend = 'log(erg/cm/s)' if radius else 'log(erg/g/s)'
            self._text(
                s = legend,
                y = 0.5,
                va = 'center',
                **text_font_kw)

    # the core figure component should become object like
    # legend, etc., with same interfaces and show/hide functions
    def show_core(self,
                  show = None,
                  draw = True):
        """
        show/hide core
        """
        if show is None:
            show = not self.showcore
        if show:
            if self.core is None:
                self.core = self.Core(self)
            else:
                self.core.show()
        else:
            if self.core is not None:
                self.core.hide()
        if draw:
            self.ax.figure.canvas.draw()

    class Core(Logged):
        """
        Class to display core.
        """
        def __init__(self, parent, coremass = None):
            self.parent = parent
            self.coremass = coremass
            self.draw()
            self.show()

        def draw(self):
            """
            Show core below substarte.

            We have to limit size as mpl plotting blows up if
            values get to big, maybe float32 ot in int32.
            """

            convdata = self.parent.convdata
            xtime = self.parent.xtime
            coremass = self.coremass

            if coremass is None:
                coremass = -1.
            center = -convdata.summ0
            if coremass > 0:
                center = np.maximum(-coremass, center)

            # TODO -
            # 1) add t-dep change of boundaries for dmacc
            # 2) add correct centre value for 'column' coord and other modes

            verts = np.ndarray((convdata.nmodels + 4, 2), np.float64)
            verts[ 0:-3,0] = xtime
            verts[-3   ,0] = xtime[-1]
            verts[-2:  ,0] = xtime[0]
            verts[ 0   ,1] = center[0]
            verts[ 1:-3,1] = center
            verts[-3:-1,1] = 0.
            verts[-1   ,1] = center[0]

            Pm = Path.MOVETO
            Pl = Path.LINETO
            Pc = Path.CLOSEPOLY
            c = [Pm] + [Pl]*(convdata.nmodels+2) + [Pc]

            path = Path(verts, codes=c)
            patch = PathPatch(
                path,
                **self.parent.subs_layer
                )
            self._patch = patch

        def hide(self):
            """
            Hide (reomve patch) core below substarte.
            """
            ax = self.parent.ax
            # ax.patches.remove(self._patch)
            ax.remove(self._patch)
            self.parent.showcore = False

        def show(self):
            """
            Show core below substarte.
            """
            ax = self.parent.ax
            if not self._patch in ax.patches:
                self.parent.ax.add_patch(self._patch)
            self.parent.showcore = True

def xrb_plot(filename, **kwargs):
    """
    routine with special defaults for XRB plots
    """
    kw = kwargs.copy()
    if not isinstance(filename, ConvData):
        c = loadconv(filename, **kw)
    else:
        c = filename
    kw.setdefault('mingain', 10)
    kw.setdefault('minloss', 7)
    kw.setdefault('showcore', True)
    kw.setdefault('decmass', True)
    kw.setdefault('column', True)
    p = ConvPlot(c, **kw)
    if kw['column']:
        ylim = [(2.e25-1.e20)/(4*np.pi*1e12), None]
    else:
        ylim = [2.e25-1.e20, None] # p.ax.get_ylim()[1]]
    p.ax.set_ylim(ylim)
    plt.draw()
    return p

# patch for matplotlib
# diff /home/alex/Python/lib/python3.5/site-packages/numpy/core/fromnumeric.py
# 2789 (numpy 1.5.1)
# <         if a.dtype == np.object:
# <             a = np.array(a, dtype=np.float)
# <     except AttributeError:
# <         pass
# <     try:
#
# this bug can be reproduced using
# > plot((1+np.array([0, 1.e-15]))*1.e27)
#
# alternatively change line 593 of ~/Python/lib/python3.5/site-packages/matplotlib/ticker.py
#
# <     locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
# ===
# >     locs = (np.asarray(_locs) - np.float(self.offset)) * 0.1 ** self.orderOfMagnitude
