import numpy as np
import matplotlib.pyplot as plt

from . import *

#######################################################################
#######################################################################
#######################################################################


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.

    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

def test(n = None):
    N = 1000
    x = np.exp(np.linspace(-2.0, 3.0, N))
    y = np.exp(np.linspace(-2.0, 2.0, N))
    x = (np.linspace(-2.0, 3.0, N))
    y = (np.linspace(-2.0, 2.0, N))

    X, Y = np.meshgrid(x, y)
    X1 = 0.5*(X[:-1,:-1] + X[1:,1:])
    Y1 = 0.5*(Y[:-1,:-1] + Y[1:,1:])

    Z1 = bivariate_normal(X1, Y1, 0.1, 0.2, 1.27, 1.11) + 100.*bivariate_normal(X1, Y1, 1.0, 1.0, 0.23, 0.72)
    Z1[Z1>0.9*np.max(Z1)] = +np.inf

    ZR = bivariate_normal(X1, Y1, 0.1, 0.2, 1.27, 1.11) + 100.*bivariate_normal(X1, Y1, 1.0, 1.0, 0.23, 0.72)
    ZG = bivariate_normal(X1, Y1, 0.1, 0.2, 2.27, 0.11) + 100.*bivariate_normal(X1, Y1, 1.0, 1.0, 0.43, 0.52)
    ZB = bivariate_normal(X1, Y1, 0.1, 0.2, 0.27, 2.11) + 100.*bivariate_normal(X1, Y1, 1.0, 1.0, 0.53, 0.92)
    ZA = bivariate_normal(X1, Y1, 0.1, 0.2, 3.27,-1.11) + 100.*bivariate_normal(X1, Y1, 1.0, 1.0, 0.23, 0.82)

    Z = np.ndarray(ZR.shape + (4,))
    Z[...,0] = ZR /ZR.max()
    Z[...,1] = ZG /ZG.max()
    Z[...,2] = ZB /ZB.max()
    Z[...,3] = np.exp(-ZA /ZA.max())


    Z = (Z*255).astype(np.uint8)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #col = plt.get_cmap('terrain_r')
    #col = get_cfunc('RGBWaves200') # need to define first, use colormap instaed
    #col = plt.get_cmap('gnuplot2')
    #col = colormap('cubehelix')
    #col = colormap('autumn')
    #col = colormap('jet')
    #col = colormap('RGBWaves', 100)
    #col = ColorBWC(mode=0)
    #col = colormap('viridis')
    col = colormap('cividis')
    #col = ColorBlindRainbow()
    #col = ColorGray()
    #col = ColorGray(gamma=(3,1,2))
    #col = IsoColorDivergeBR()
    #col = ColorMapList(['Orange', 'Gold', 'Salmon', 'Tomato', 255], gamma = 2)
    #col = ColorSolid(.5)
    #col = ColorJoin((ColorGray(.5), BlendInFilter(ColorBlindRainbow(gamma = .5), frac=0.25)), .3)
    #col = ColorJoin((ColorGray(-.5), BlendInFilter(ColorBlindRainbow(gamma = .5), frac=-0.25), IsoColorDivergeBR()), blend=(.5,.1))
    #col = ColorJoin((ColorGray(.5), ColorBlindRainbow(gamma = .5), 'green'), blend = 0.1, method = 'linear')
    #col = ColorBlindRainbow(gamma = .5)
    #col = FilterColorBlindRed(IsoColors('BlindRainbow', 9))
    #col = IsoColorDivergeBWR(n)
    #col = IsoColorRainbow15()
    #col = FilterColorGray(IsoColorRainbow21())
    #col = IsoColorLights('BlindRainbow')
    #col = IsoColorRainbow(6)
    #col = ColorBlend((BlendInFilter(ColorBlindRainbow(gamma = .5), frac=0.25), ColorGray(.5)))
    #col = ColorBlendBspline((ColorGray(.5), WaveFilter(ColorBlindRainbow(), nwaves = 20)),)
    #col = ColorBlend((ColorMapList(['Green']), ColorMapList(['Red'])))
    #col = ColorBlendBspline(('red', 'green', 'yellow', 'blue'), k=2)
    #col = ColorDiverging(('#D00000','#808080','#0000D0'))
    #col = ColorCircle(func = lambda x: x**10)
    #col = ColorCircle(model = 'HCL', init = [0, 1, .5])
    #col = ColorCircle(model = 'LChuv', init = [100, 100, .5])
    #col = get_cmap('autumn')
    #col = ColorBlend((ColorGray(reverse=False), HueRotateFilter(ColorBlendWave('Gold', 'Purple', nwaves = 20), angle = 180)))
    #col = ShadeColor(gamma=1/2.2)
    #col = ShadeColor(gamma=1, model='RGB')

    #input filter
    #col = ColorScale(col, lambda x : (np.log((x)+1)/(np.exp(-3)-1))
    #col = ColorScaleGamma(col, -1/2.2)
    #col = ColorScaleExp(col, 3)
    #col = ColorScaleLog(col, 3)

    # levels
    #col = IsoColors(col, 12)

    # COLOR FILTERS
    #col = FilterColorGray(col)
    #col = WaveFilter(col, nwaves = 20, property = 'H')
    #col = WaveFilter(col, nwaves = 20, property = 'I', model = 'YIQ')
    #col = WaveFilter(col, nwaves = 20, index = 2, method = 2, model = 'LChuv')
    #col = WaveFilter(col, nwaves = 20, index = 0, method = 0, model = 'LChuv')
    #col = WaveFilter(col, nwaves = 20, property = 'V', model = 'HSV')
    #col = WaveFilter(col, nwaves = 20, property = 'S', model = 'HSV')
    #col = WaveFilter(col, nwaves = 20, property = 'C', model = 'CMY')
    #col = WaveFilter(col, nwaves = 20)
    #col = BlendInFilter(col, frac=-0.5)
    #col = FilterColorModel(col, model = 'sRGB')
    #col = Reverse(col) # why does it invert neg inf when used with WaveFilter?
    #col = FilterColorInvert(col)
    #col = FilterColorHueRotate(col, model = 'HCL', angle = 60)
    #col = FilterColorGamma(col, gamma = (1,2,3,4))
    #col = AlphaFilter(col, func = lambda x: 0.5*(1+np.sin(100*x)))
    #col = AlphaFilter(col, func = lambda x: x**2)
    #col = FuncBackgroundFilter(col, func = lambda x: x**2)
    #col = FuncBackgroundFilter(col, func = lambda x: 1 -2*np.abs(x-0.5))
    #col = ColorFilterColor(col, color = 'Pink')
    #col = HueRotateFilter(col, func = lambda x: 0.5 * np.sin(40*np.pi*x))
    #col = ColorFilterColor(col, color = 'Pink', reverse = True, method = 'min', clip = False)
    #col = ColorXFilterColor(col, color = 'Pink', reverse = False, method = 'vec')
    #col = ColorBlendBspline(('red','green','Gold','blue', ColorScale(col, lambda x: x*5-4)), k=2)
    #col = ComponentFilter(col, func = lambda x : 1-x**2)
    #col = FilterColorTransparent(col, transparent = ('#00007f', '#7f7fff'))
    #col = SetBackgroundFilter(col, background = '#FF0000')
    #col = FilterColorGraysRGB(col)

    # YFILTERS
    #Z2 = (x[1:] + x[-1])[None, :] * (y[1:] + y[-1])[:, None]
    #Z2 /= np.max(Z2)
    #col = FuncAlphaYFilter(col, data=Z2)


    # TEST INVERSE
    # model = 'Luv'
    # col = FilterColorModel(col, model = color_model(model).inverse())
    # col = FilterColorModel(col, model = color_model(model))


    # VISION IMPARED COLOR FILTERS
    #col = FilterColorBlindRed(col)
    #col = FilterVisCheck(col, 'tritanope')
    #col = FilterColorBlindness(col, 'achroma')
    #col = FilterColorBlindness(col, 'protan')
    #col = FilterColorBlindness(col, 'tritan')
    #col = FilterColorBlindness(col, 'deutan')


    #i = ax.pcolorfast(x, y, Z1, cmap = col, alpha=1.)
    #i = ax.pcolorfast(x, y, Z1, cmap = col)
    #Z1 = col(Z1, bytes = True)
    #i = pcolorimage(ax, x, y, Z1)

    # from matplotlib.image import PcolorImage
    i = PcolorImage(ax, x, y, Z)
    ax.images.append(i)
    # xl, xr, yb, yt = x[0], x[-1], y[0], y[-1]
    # ax.update_datalim(np.array([[xl, yb], [xr, yt]]))
    # ax.autoscale_view(tight=True)
    plt.colorbar(i)
    plt.show()
    return i

def test3():
    N = 1000
    x = (np.linspace(-2.0, 3.0, N))
    y = (np.linspace(-2.0, 2.0, N))

    X, Y = np.meshgrid(x, y)
    X1 = 0.5*(X[:-1,:-1] + X[1:,1:])
    Y1 = 0.5*(Y[:-1,:-1] + Y[1:,1:])

    from matplotlib.mlab import bivariate_normal
    Z1 = bivariate_normal(X1, Y1, 0.1, 0.2, 1.27, 1.11) + 100.*bivariate_normal(X1, Y1, 1.0, 1.0, 0.23, 0.72)


    Z1[Z1>0.9*np.max(Z1)] = +np.inf

    cdict = {'red':   [(0.0,  1.0, 1.0),
                       (1.0,  1.0, 1.0)],

             'green': [(0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)],

             'blue':  [(0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)],

             'alpha': [(0.0,  0.0, 0.0),
                       (1.0,  1.0, 1.0)],
             }

    # this is a color maps showing how the resukt should look like
    cdictx = {'red':   [(0.0,  1.0, 1.0),
                       (1.0,  1.0, 1.0)],

             'green': [(0.0,  1.0, 1.0),
                       (1.0,  0.0, 0.0)],

             'blue':  [(0.0,  1.0, 1.0),
                       (1.0,  0.0, 0.0)],

             }

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    col = mpl_colors.LinearSegmentedColormap('test', cdict)
    i = ax.pcolorfast(x, y, Z1, cmap = col)
    plt.colorbar(i)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    col = mpl_colors.LinearSegmentedColormap('testx', cdictx)
    i = ax.pcolorfast(x, y, Z1, cmap = col)
    plt.colorbar(i)

    plt.show()



def test2(n = 3):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    x = np.linspace(0,1,1000)
    c = IsoColorBlind(n)
    #c = IsoColorDivergeBWR(n)
    #c = isoshadecolor(n)
    #c = FilterColorBlindRed(IsoColorLights(ColorBlindRainbow, n=2))
    #c = IsoColorRainbow(n)
    #c = IsoGraySave(n)
    #c = FilterColorBlindRed(c)
    #c = FilterColorGray(c)
    #c = WaveFilter(c)
    for i in range(len(c)):
        ax.plot(x,x**(i+1),c = c[i])

def _test_get_color_index(color):
    if color is None:
        color = 'white'
    if color == 'white':
        ii = slice(None)
    elif color == 'blue':
        ii = slice(2, None, 3)
    elif color == 'green':
        ii = slice(1, None, 3)
    elif color == 'red':
        ii = slice(0, None, 3)
    elif color == 'cyan':
        ii = slice(1, 3)
    elif color == 'yellow':
        ii = slice(0, 2)
    elif color == 'purple':
        ii = slice(0, 3, 2)
    else:
        raise Exception('Unknown color.')
    return ii

def _test_set_pattern(x, level, ii, sat):
    if level == 1/2:
        x[::2,::2,ii] = sat
        x[1::2,1::2,ii] = sat
    elif level == 1/4:
        x[::2,::4,ii] = sat
        x[1::2,2::4,ii] = sat
    elif level == 3/4:
        x[:,:,ii] = sat
        x[::2,::4,ii] = 0
        x[1::2,2::4,ii] = 0
    elif level == 1/3:
        x[::3,::3,ii] = sat
        x[2::3,1::3,ii] = sat
        x[1::3,2::3,ii] = sat
    elif level == 2/3:
        x[:,:,ii] = sat
        x[::3,::3,ii] = 0
        x[2::3,1::3,ii] = 0
        x[1::3,2::3,ii] = 0
    else:
        jj = np.random.random(x.shape[:2]) < level
        x[jj,ii] = sat

def test_gamma(gamma=1, level = 0.5, sat = 1, color = None):
    """
    test gamma scale

    Parameters
    ----------

    gamma:
        gamma to use, default = 1

    level:
        pattern fill fraction, default = 0.5

    sat:
        saturation of fill pattern, default = 1

    color:
        string for color, defaul = 'white'
    """
    fig = plt.figure(facecolor='k')
    dx, dy = [int(x) for x in fig.get_window_extent().size]
    x = np.zeros((dy, dx//2, 3))
    y = np.zeros((dy, dx - dx//2, 3))
    ii = _test_get_color_index(color)
    y[:,:,ii]=((1-np.arange(dy)/(dy-1))**(1/gamma)).reshape((dy,1,1))
    _test_set_pattern(x, level, ii, sat)
    fig.figimage(x)
    fig.figimage(y, xo = dx//2, yo = 0)


def test_gamma2(color = 'white', level = 0.5, sat = 1, bits = 8):
    """
    Determine gamma scale.

    Parameters
    ----------

    gamma:
        gamma to use, default = 1

    level:
        pattern fill fraction, default = 0.5

    sat:
        saturation of fill pattern, default = 1

    color:
        string for color, defaul = 'white'

    bits:
        bits used for level, default = 8

    Usage
    -----

    Use keys <left> / <right> to increase / decrease bit level by one.
    """
    scale = 2**bits - 1
    l = int(0.5**(0.5) * scale)
    ii = _test_get_color_index(color)
    fig = plt.figure(facecolor = 'k')
    dx, dy = [int(x) for x in fig.get_window_extent().size]
    x = np.zeros((dy, dx, 3))
    y = np.zeros((dy//3, dx//3, 3))
    _test_set_pattern(x, level, ii, sat)
    y[:,:,ii] = l / scale
    img_x = fig.figimage(x)
    img_y = fig.figimage(y, xo = dx//3, yo = dy//3)

    def onkeypress(event):
        nonlocal l, img_y
        update = False
        if event.key == 'left':
            l = min(scale, l + 1)
            y[:,:,ii]=l/scale
            update = True
        if event.key == 'right':
            l = max(0, l - 1)
            y[:,:,ii]=l/scale
            update = True
        if update:
            fig.images.remove(img_y)
            img_y = fig.figimage(y, xo = dx//3, yo = dy//3)
            fig.show()
            gamma = np.log(level) / np.log(l / (scale * sat))
            #print(f'level: {l}, gamma={gamma:5.3f}')
            print('level: {}, gamma={:5.3f}'.format(l, gamma))
    fig.canvas.mpl_connect('key_press_event', onkeypress)

def test_gamma3(color = 'white', level = 0.5, sat = 1, bits = 8):
    """
    Determine gamma scale with individual color adjustment.

    Parameters
    ----------

    gamma:
        gamma to use, default = 1

    level:
        pattern fill fraction, default = 0.5

    sat:
        saturation of fill pattern, default = 1

    color:
        string for color, defaul = 'white'

    bits:
        bits used for level, default = 8

    Usage
    -----

    Use keys <left> / <right> to increase / decrease bit level by one.
    Use keys <R> / <r> to increase / decrease bit level of *red* by one.
    Use keys <G> / <g> to increase / decrease bit level of *green* by one.
    Use keys <B> / <b> to increase / decrease bit level of *blue* by one.
    """
    scale = 2**bits - 1
    ii = _test_get_color_index(color)
    l = np.zeros(3, dtype = np.int)
    l[ii] = int(0.5**(0.5) * scale)
    fig = plt.figure(facecolor = 'k')
    dx, dy = [int(x) for x in fig.get_window_extent().size]
    x = np.zeros((dy, dx, 3))
    y = np.zeros((dy//3, dx//3, 3))
    _test_set_pattern(x, level, ii, sat)
    jj = (np.newaxis, np.newaxis, slice(None))
    y[:,:,:] = (l / scale)[jj]
    img_x = fig.figimage(x)
    img_y = fig.figimage(y, xo = dx//3, yo = dy//3)

    def onkeypress(event):
        nonlocal l, img_y
        update = False
        keys = {
            'r' : (0, -1),
            'R' : (0, +1),
            'g' : (1, -1),
            'G' : (1, +1),
            'b' : (2, -1),
            'B' : (2, +1),
            'left' : (slice(None), +1),
            'right' : (slice(None), -1),
            }
        if event.key in keys:
            k = keys[event.key]
            l[k[0]] = np.maximum(np.minimum(l[k[0]] + k[1], scale), 0)
            y[:,:,:]=(l/scale)[jj]
            fig.images.remove(img_y)
            img_y = fig.figimage(y, xo = dx//3, yo = dy//3)
            fig.show()
            gamma = np.log(level) / np.log(l / (scale * sat))
            print(l, gamma)
    fig.canvas.mpl_connect('key_press_event', onkeypress)



from matplotlib.image import PcolorImage
def pcolorimage(ax, x, y, Z):
    """
    Make a PcolorImage based on Z = (x, y, 4) [byte] array

    This is to fix an omission ('bug') in the current (as of this
    writing) version of MatPlotLib.  I may become superfluous in the
    future.
    """
    img = PcolorImage(ax, x, y, Z)
    ax.images.append(img)
    xl, xr, yb, yt = x[0], x[-1], y[0], y[-1]
    ax.update_datalim(np.array([[xl, yb], [xr, yt]]))
    ax.autoscale_view(tight=True)
    return img

# lots of color data: http://www.cvrl.org/
