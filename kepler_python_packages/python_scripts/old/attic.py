class MultiLoop(object):
    """
    Provides multi_loop method to loop over all iterable parameters
    except strings.

    Use:
        class X( ..., MultiLoop, ...)

    Call:
        self.multi_loop(self.method_to_run, *args, **kwargs)

    LIMITATIONS:
        currently levels for non-dict are not preserved, i.e.,
        [[1,2],3] will have the same result as [1,2,3]
        In contast, dict[1] in second level will be preserved.
    """
    def multi_loop(self, method, *args, **kwargs):
        """
        Loops over all iterable parameters except strings.
        """
        kwargs_new = dict(kwargs)
        args_new = list(args)
        for iv,v in enumerate(args):
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    args_new[iv] = {k:i}
                    self.multi_loop(method, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    args_new[iv] = i
                    self.multi_loop(method, *args_new, **kwargs_new)
                return
        for kw,v in kwargs.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    kwargs_new[kw] = {k:i}
                    self.multi_loop(method, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    kwargs_new[kw] = i
                    self.multi_loop(method, *args_new, **kwargs_new)
                return
        method(*args, **kwargs)

class MultiLoopOne(object):
    """
    Provides multi_loop method to loop over all iterable parameters
    except strings.

    Use:
        class X( ..., MultiLoop, ...)

    Call:
        self.multi_loop(self.method_to_run, *args, **kwargs)

    LIMITATIONS:
        currently levels for non-dict are not preserved, i.e.,
        [[1,2],3] will have the same result as [1,2,3]
        In contast, dict[1] in second level will be preserved.
    """
    class _multi_loop_container(object):
        def __init__(self, item):
            self.item = item
    def multi_loop(self, method, *args, **kwargs):
        """
        Loops over all iterable parameters except strings.
        """
        kwargs_new = dict(kwargs)
        args_new = list(args)
        for iv,v in enumerate(args):
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    args_new[iv] = {k:i}
                    self.multi_loop(method, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    args_new[iv] = self._multi_loop_container(i)
                    self.multi_loop(method, *args_new, **kwargs_new)
                return
        for kw,v in kwargs.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    kwargs_new[kw] = {k:i}
                    self.multi_loop(method, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    kwargs_new[kw] = self._multi_loop_container(i)
                    self.multi_loop(method, *args_new, **kwargs_new)
                return
        # get rid of containers
        for iv,v in enumerate(args):
            if isinstance(v, self._multi_loop_container):
                if isinstance(args, tuple):
                    args = list(args)
                args[iv] = v.item
        for kw,v in kwargs.items():
            if isinstance(v, self._multi_loop_container):
                kwargs[kw] = v.item
        method(*args, **kwargs)

class MultiLoopCollect(object):
    """
    Provides multi_loop method to loop over all iterable parameters
    except strings and retuns list of results.

    Use:
    class X( ..., MultiLoop, ...):

    Call:
    x = self.multi_loop(self.method_to_run, *args, **kwargs)
    """
    def multi_loop_collect(self, method, *args, **kwargs):
        """
        Loops over all iterable parameters except strings.
        """
        kwargs_new = dict(kwargs)
        args_new = list(args)
        result = []
        for iv,v in enumerate(args):
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    args_new[iv] = {k:i}
                    result += self.multi_loop_collect(method, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    args_new[iv] = i
                    result += self.multi_loop_collect(method, *args_new, **kwargs_new)
                return result
        for kw,v in kwargs.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    kwargs_new[kw] = {k:i}
                    result += self.multi_loop_collect(method, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    kwargs_new[kw] = i
                    result += self.multi_loop_collect(method, *args_new, **kwargs_new)
                return result
        return [method(*args, **kwargs)]




class MultiLoop(object):
    """
    Provides multi_loop method to loop over all iterable parameters
    except strings.

    Use:
        class X( ..., MultiLoop, ...)

    Call:
        self.multi_loop(self.method_to_run, *args, **kwargs)

    Parameters:
        a keyword parameter loop_descend decides what to do with nested
        iterables

    LIMITATIONS:
        In contast, dict[1] in second level will be preserved.
    """
    class _multi_loop_container(object):
        def __init__(self, item):
            self.item = item
    def multi_loop(self, method, *args, **kwargs):
        """
        Loops over all iterable parameters except strings.
        """
        kwargs_new = dict(kwargs)
        args_new = list(args)
        descend = kwargs_new.setdefault('loop_descend', False)
        del kwargs_new['loop_descend']
        result = []
        for iv,v in enumerate(args):
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    args_new[iv] = {k:i}
                    result += self.multi_loop(method, *args_new, loop_descend = descend, **kwargs_new)
                return result
            if isinstance(v, Iterable):
                for i in v:
                    if descend:
                        args_new[iv] = i
                    else:
                        args_new[iv] = self._multi_loop_container(i)
                    result += self.multi_loop(method, *args_new, loop_descend = descend, **kwargs_new)
                return result
        for kw,v in kwargs.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    kwargs_new[kw] = {k:i}
                    result += self.multi_loop(method, *args_new, loop_descend = descend, **kwargs_new)
                return result
            if isinstance(v, Iterable):
                for i in v:
                    if descend:
                        kwargs_new[kw] = i
                    else:
                        kwargs_new[kw] = self._multi_loop_container(i)
                    result += self.multi_loop(method, *args_new, loop_descend = descend, **kwargs_new)
                return result
        # get rid of containers
        if not descend:
            for iv,v in enumerate(args_new):
                if isinstance(v, self._multi_loop_container):
                    args_new[iv] = v.item
            for kw,v in kwargs_new.items():
                if isinstance(v, self._multi_loop_container):
                    kwargs_new[kw] = v.item
        return [method(*args_new, **kwargs_new)]


def loopedmethod(method, *args, **kwargs):
    """
    Decorator to compute a looped method.

    Use:
    @loopedmethod
    method_to_loop

    Call:
    self.method_to_loop(*args,**kwargs)
    """
    def looped_method(self, *args, **kwargs):
        """
        Loop over all Iterables in *args and **kwargs except strings.
        """
        kwargs_new = dict(kwargs)
        args_new = list(args)
        for iv,v in enumerate(args):
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    args_new[iv] = {k:i}
                    looped_method(self, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    args_new[iv] = i
                    looped_method(self, *args_new, **kwargs_new)
                return
        for kw,v in kwargs.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    kwargs_new[kw] = {k:i}
                    looped_method(self, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    kwargs_new[kw] = i
                    looped_method(self, *args_new, **kwargs_new)
                return
        method(self, *args, **kwargs)
    looped_method.__dict__.update(method.__dict__)
    looped_method.__dict__['method'] = method.__name__
    if method.__doc__ is not None:
        looped_method.__doc__ = method.__doc__ + '\n' + looped_method.__doc__
    looped_method.__name__ = method.__name__
    looped_method.__module__ = getattr(method, '__module__')
    return looped_method


def looponemethod(method):
    """
    Decorator to compute a looped method.

    Use:
    @loopedmethod
    method_to_loop

    Call:
    self.method_to_loop(*args,**kwargs)
    """
    class _multi_loop_container(object):
        def __init__(self, item):
            self.item = item
    def looped_method(self, *args, **kwargs):
        """
        Loop over all Iterables in *args and **kwargs except strings.
        """
        kwargs_new = dict(kwargs)
        args_new = list(args)
        for iv,v in enumerate(args):
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    args_new[iv] = {k:i}
                    looped_method(self, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    args_new[iv] = _multi_loop_container(i)
                    looped_method(self, *args_new, **kwargs_new)
                return
        for kw,v in kwargs.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    kwargs_new[kw] = {k:i}
                    looped_method(self, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    kwargs_new[kw] = _multi_loop_container(i)
                    looped_method(self, *args_new, **kwargs_new)
                return
        # get rid of containers
        for iv,v in enumerate(args):
            if isinstance(v, _multi_loop_container):
                if isinstance(args, tuple):
                    args = list(args)
                args[iv] = v.item
        for kw,v in kwargs.items():
            if isinstance(v, _multi_loop_container):
                kwargs[kw] = v.item
        method(self, *args, **kwargs)
    looped_method.__dict__.update(method.__dict__)
    looped_method.__dict__['method'] = method.__name__
    if method.__doc__ is not None:
        looped_method.__doc__ = method.__doc__ + '\n' + looped_method.__doc__
    looped_method.__name__ = method.__name__
    looped_method.__module__ = getattr(method, '__module__')
    return looped_method

def loopedcollectmethod(method, *args, **kwargs):
    """
    Decorator to compute a looped method and retruns list of restults.

    Use:xs
    @loopedcollectmethod
    method_to_loop

    Call:
    x = self.method_to_loop(*args,**kwargs)
    """
    def looped_collect_method(self, *args, **kwargs):
        """
        Loop over all Iterables in *args and **kwargs except strings.
        """
        kwargs_new = dict(kwargs)
        args_new = list(args)
        result = []
        for iv,v in enumerate(args):
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    args_new[iv] = {k:i}
                    result += looped_collect_method(self, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    args_new[iv] = i
                    result += looped_collect_method(self, *args_new, **kwargs_new)
                return result
        for kw,v in kwargs.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                if len(v) <= 1:
                    continue
                for k,i in v.items():
                    kwargs_new[kw] = {k:i}
                    result += looped_collect_method(self, *args_new, **kwargs_new)
                return
            if isinstance(v, Iterable):
                for i in v:
                    kwargs_new[kw] = i
                    result += looped_collect_method(self, *args_new, **kwargs_new)
                return result
        return [method(self, *args, **kwargs)]
    looped_collect_method.__dict__.update(method.__dict__)
    looped_collect_method.__dict__['method'] = method.__name__
    if method.__doc__ is not None:
        looped_collect_method.__doc__ = (
            method.__doc__ +
            '\n' +
            looped_collect_method.__doc__)
    looped_collect_method.__name__ = method.__name__
    looped_collect_method.__module__ = getattr(method, '__module__')
    return looped_collect_method


def loopmethod(descend = False):
    """
        Decorator to compute a looped method.

        Use:
        @loopmethod(descend)
        method_to_loop

        If descend is False, stope at first level, otherwise descend
        down nested lists, sets, and tuples.

        Call:
        self.method_to_loop(*args,**kwargs)

        Returns list of results.

        TODO: could add "str" parameter to resolve strings
    """
    def loop_method(method):
        """
        Decorator to compute a looped method.

        Use:
        @loopedmethod
        method_to_loop

        Call:
        self.method_to_loop(*args,**kwargs)
        """
        class _container(object):
            def __init__(self, item):
                self.item = item
        def looped_method(self, *args, **kwargs):
            """
            Loop over all Iterables in *args and **kwargs except strings.
            """
            kwargs_new = dict(kwargs)
            args_new = list(args)
            result = []
            for iv,v in enumerate(args):
                if isinstance(v, str):
                    continue
                if isinstance(v, dict):
                    if len(v) <= 1:
                        continue
                    for k,i in v.items():
                        args_new[iv] = {k:i}
                        result += looped_method(self, *args_new, **kwargs_new)
                    return result
                if isinstance(v, Iterable):
                    for i in v:
                        if descend:
                            args_new[iv] = i
                        else:
                            args_new[iv] = _container(i)
                        result += looped_method(self, *args_new, **kwargs_new)
                    return result
            for kw,v in kwargs.items():
                if isinstance(v, str):
                    continue
                if isinstance(v, dict):
                    if len(v) <= 1:
                        continue
                    for k,i in v.items():
                        kwargs_new[kw] = {k:i}
                        result += looped_method(self, *args_new, **kwargs_new)
                    return result
                if isinstance(v, Iterable):
                    for i in v:
                        if descend:
                            kwargs_new[kw] = i
                        else:
                            kwargs_new[kw] = _container(i)
                        result += looped_method(self, *args_new, **kwargs_new)
                    return result
            # get rid of containers
            if not descend:
                for iv,v in enumerate(args):
                    if isinstance(v, _container):
                        if isinstance(args, tuple):
                            args = list(args)
                        args[iv] = v.item
                for kw,v in kwargs.items():
                    if isinstance(v, _container):
                        kwargs[kw] = v.item
            return [method(self, *args, **kwargs)]
        looped_method.__dict__.update(method.__dict__)
        looped_method.__dict__['method'] = method.__name__
        if method.__doc__ is not None:
            looped_method.__doc__ = method.__doc__ + '\n' + looped_method.__doc__
        looped_method.__name__ = method.__name__
        looped_method.__module__ = getattr(method, '__module__')
        return looped_method
    return loop_method


from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform

class JustScale(ScaleBase):
    """
    just linear scale
    """
    name = 'scale'

    class ScaleTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, scale):
            Transform.__init__(self)
            self.scale = np.float64(scale)

        def transform(self, a):
            return a * self.scale

        def inverted(self):
            return JustScale.InvertedScaleTransform(
                self.scale)

    class InvertedScaleTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, scale):
            Transform.__init__(self)
            self.scale = np.float64(scale)

        def transform(self, a):
            return a / self.scale

        def inverted(self):
            return JustScale.ScaleTransform(
                self.scale)

    def __init__(self, axis, **kwargs):
        """
        scale 1.0
        """
        if axis.axis_name == 'x':
            scale = kwargs.pop('scalex', 1.0)
        else:
            scale = kwargs.pop('scaley', 1.0)

        self._transform = self.ScaleTransform(scale)

        assert scale != 0.0
        self.scale = scale

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        symmetrical log scaling.
        """
        from matplotlib.ticker import AutoLocator, ScalarFormatter
        from matplotlib.ticker import NullLocator, NullFormatter

        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def get_transform(self):
        """
        Return a :class:`ScaleTransform` instance.
        """
        return self._transform

register_scale(JustScale)

from multiprocessing import Pool

def process_layer(layer):
    layer.process()
    return (layer.color, layer.path)

class LayerProcess(object):
    def __init__(self,
                 layer = None,
                 quant = None,
                 level = None,
                 xtime = None,
                 data = None,
                 radius = None,
                 color = None,
                 clayer = None):
        self.data = data
        self.radius = radius
        self.layer = layer
        self.quant = quant
        self.level = level
        self.xtime = xtime
        self.clayer = clayer
        self.color = color
        self.patch = None

    def process(self):
        print('processing level {:d}'.format(self.level))
        if self.layer is None:
            self.layer = extract_layer(self.data, self.quant, self.level, self.radius)
            layer = Layer(*self.layer)
            self.path = layer.poly_path(self.xtime)
        elif thread == 1:
            pool = Pool()
            ll = list()
            for i in range(min_gain_level,max_gain_level+1):
                if False:
                    lc = None
                    ld = convdata.data
                else:
                    lc = convdata.extract_layer(quant,level=i)
                    ld = None
                task = LayerProcess(
                    layer = lc,
                    radius = layer_radial,
                    data = ld,
                    quant = layer_index,
                    xtime = xtime,
                    color = self.gain_layer,
                    level = i)
                ll.append(task)

            ires = pool.imap_unordered(process_layer, ll)

            ll = list()
            for i in range(-min_loss_level,-max_loss_level-1,-1):
                lc = convdata.extract_layer(quant,level=i)
                ld = None
                task = LayerProcess(
                    layer = lc,
                    radius = layer_radial,
                    data = ld,
                    quant = layer_index,
                    xtime = xtime,
                    color = self.loss_layer,
                    level = i)
                ll.append(task)

            self.add_timer('thread')
            jres = pool.imap_unordered(process_layer, ll)
            pool.close()
            pool.join()
            for l in ires:
                patch = PathPatch(l[1],**l[0])
                self.ax.add_patch(patch)
            for l in jres:
                patch = PathPatch(l[1],**l[0])
                self.ax.add_patch(patch)

            self.logger.info('threads finished in {:s}'.format(time2human(self.finish_timer('thread'))))

    # def _show_models_save(self,
    #                  convdata,
    #                  xtime):
    #     self.add_timer('show_models')
    #     self.logger.info('Plotting models')

    #     figbox = self.ax.figbox
    #     base = 1
    #     height = (1 - figbox.y1) / figbox.height
    #     ytick0 = base + 0.4 * height
    #     colors = ['black','red','green','blue']
    #     trans = blended_transform_factory(
    #         self.ax.transData,
    #         self.ax.transAxes)
    #     bbox = Bbox([[0,base],[1,base+height]])
    #     cbox = TransformedBbox(bbox,
    #                            self.ax.transAxes)
    #     for i, data in enumerate(convdata.data):
    #         mag = 0
    #         val = data.ncyc
    #         while (val % 10) == 0:
    #             mag += 1
    #             val //= 10
    #         half = int((val % 5) == 0)
    #         line = Line2D(np.tile(xtime[i+1],2),
    #                       (ytick0 + np.array([0,1])*
    #                        (3 + 1.5*mag + half) * 0.05 * height),
    #                       transform = trans,
    #                       color = colors[mag % len(colors)],
    #                       linewidth = 0.1 * (1 + mag + half))
    #         self.ax.add_line(line)
    #         line.set_clip_box(cbox)
    #     fp = FontProperties()
    #     fp.set_size(0.71 * fp.get_size())
    #     text = Text(x = 0.5,
    #                 y = 1+height,
    #                 text = 'models',
    #                 color = 'black',
    #                 clip_on = False,
    #                 ha = 'center',
    #                 va = 'top',
    #                 fontproperties = fp,
    #                 transform = self.ax.transAxes)
    #     self.ax.add_artist(text)
    #     self.logger.info('Plotting models finished in {:s}'.format(time2human(self.finish_timer('show_models'))))



class CrazyScale(ScaleBase):
    """
    Cracy Scale
    """
    name = 'crazyscale'

    class ScaleTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, scale):
            Transform.__init__(self)
            self.scale = np.float64(scale)

        def transform_non_affine(self, a):
            return np.minimum(np.tanh(a * self.scale), 0.9999)

        def inverted(self):
            return CrazyScale.InvertedScaleTransform(
                self.scale)

    class InvertedScaleTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, scale):
            Transform.__init__(self)
            self.scale = np.float64(scale)

        def transform_non_affine(self, a):
            masked = ma.masked_where((a > 0.9999), a)
            print(max(a))
            if masked.mask.any():
                return ma.arctanh(masked) / self.scale
            else:
                return np.arctanh(a) / self.scale

        def inverted(self):
            return CrazyScale.ScaleTransform(
                self.scale)

    def __init__(self, axis, **kwargs):
        """
        scale 1.0
        """
        scale = kwargs.pop('power', 1.0)
        if axis.axis_name == 'x':
            scale = kwargs.pop('powerx', scale)
        else:
            scale = kwargs.pop('powery', scale)

        print('scale = {}'.format(scale), axis.axis_name)

        self._transform = self.ScaleTransform(scale)

        assert scale != 0.0
        self.scale = scale

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        symmetrical log scaling.
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0, vmin), min(vmax, 0.5)


    def get_transform(self):
        """
        Return a :class:`ScaleTransform` instance.
        """
        print('getting transform')
        return self._transform


register_scale(CrazyScale)
