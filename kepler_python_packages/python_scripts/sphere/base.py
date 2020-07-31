"""
base classes for visuaisations
"""

import time
import functools
import types
import re
import sys

import numpy as np

import vtk

from movie.vtkbase import VtkBase
from movie import MovieWriter, NoMovieWriter, BaseMovieWriter
from movie import ProcessorChain, ImgLogoProcessor, FontProcessor, RotFontProcessor


NODEFAULT = object()

class ResolveDefaults(object):

    _resolve_mode = 'pop'
    _resolve_debug = False
    _resolve_pattern = re.compile(r'^_(?:[^_]+.*)?[^_]$')

    def __init__(self, *args, **kwargs):
        self.resolve_defaults(kwargs)

    def resolve_defaults(self, kwargs, resolve_mode = None, resolve_debug = None):
        if resolve_mode is None:
            resolve_mode = getattr(self, 'resolve_mode', None)
        if resolve_mode is None:
            resolve_mode = kwargs.get('resolve_mode', self._resolve_mode)
        if resolve_debug is None:
            resolve_debug = getattr(self, 'resolve_debug', None)
        if resolve_debug is None:
            resolve_debug = kwargs.get('resolve_debuf', self._resolve_debug)
        cls = self.getcallerclass()
        mro = cls.__mro__
        items = set(k for k,v in mro[0].__dict__.items() if
                 (len(self._resolve_pattern.findall(k)) == 1 and
                   not isinstance(v, types.FunctionType)))
        for m in mro[1:]:
            for k in m.__dict__.keys():
                if resolve_debug and k in items:
                    print(f'[DEBUG] ELIMINATE default from {m.__name__}.{k} for {cls.__name__}')
                items -= {k}
        if resolve_mode == 'pop':
            default = self.popdefault
        elif resolve_mode == 'get':
            default = self.getdefault
        elif resolve_mode == 'set':
            default = self.setdefault
        else:
            default = self.default
        for i in items:
            if resolve_debug:
                print(f'[DEBUG] setting default from {cls.__name__}.{i}')
            item = i[1:]
            default(kwargs, item)
            # set dict field from kwargs 'attr.subkey'
            if default == self.default:
                continue
            try:
                attr = getattr(self, item)
            except:
                continue
            if not isinstance(attr, dict):
                continue
            for k,v in kwargs.items():
                if k.startswith(item + '.'):
                    subkey = k[len(item)+1:]
                    attr[subkey] = v
            # TODO - recursevely resolve dict, lists, ...

    def getcallerclass(self):
        frame = sys._getframe().f_back.f_back
        mro = self.__class__.__mro__
        code = frame.f_code
        name = code.co_name
        for m in mro:
            d = m.__dict__
            if name in d:
                fn = d[name]
                if fn.__code__ is code:
                    return m
        raise Exception('Cannot resolve caller class.  Caller needs to be method of derived class.')

    def default(self, kwargs, name):
        val = getattr(self, f'_{name}')
        if val is NODEFAULT:
            return kwargs.get(name, val)
        return kwargs.setdefault(name, val)

    def setdefault(self, kwargs, name):
        val = getattr(self, f'_{name}')
        if val is NODEFAULT:
            val = kwargs.get(name, val)
            if val is NODEFAULT:
                return
        else:
            val = kwargs.setdefault(name, val)
        setattr(self, name, val)

    def getdefault(self, kwargs, name):
        val = kwargs.get(name, getattr(self, f'_{name}'))
        if val is NODEFAULT:
            return
        setattr(self, name, val)

    def popdefault(self, kwargs, name):
        val = kwargs.pop(name, getattr(self, f'_{name}'))
        if val is NODEFAULT:
            return
        setattr(self, name, val)

class SceneItem(ResolveDefaults):
    def __init__(self):
        self.actors = list()
        self.items = list()

    def clear(self):
        self.clear_items()
        self.clear_actors()

    def clear_items(self):
        self.items.clear()

    def clear_actors(self):
        self.actors.clear()

    def replace_actors(self, actors, static = False):
        self.actors.clear()
        self.add_actors(actors, static)

    def add_item(self, item):
        if not hasattr(self, 'items'):
            self.items = list()
        self.items.append(item)

    def add_actor(self, actor, static = False):
        if not hasattr(self, 'actors'):
            self.actors = list()
        self.actors.append((actor, static))

    def add_items(self, items):
        self.items.extend(list(items))

    def add_actors(self, actors, static = False):
        self.actors.extend(list((a, static) for a in actors))

    def get_actors(self, static = None):
        actors = list()
        for item in self.get_items():
            ax = item.actors
            for (a, s) in ax:
                if static is None or static == s:
                    actors.append(a)
        return actors

    def get_items(self):
        items = [self]
        if hasattr(self, 'items'):
            for item in self.items:
                items.extend(list(item.get_items()))
        return items

    def draw_static(self):
        pass

    def draw(self, time):
        pass

    def checked_draw(self, time):
        a = self.actors.copy()
        self.draw(time)
        l = len(a)
        assert a[:] == self.actors[:l]
        for (x, s) in self.actors[l:]:
            assert s == False

    def checked_draw_static(self):
        a = self.actors.copy()
        self.draw_static()
        l = len(a)
        assert a[:] == self.actors[:l]
        for (x, s) in self.actors[l:]:
            assert s == True

    def remove_actors(self, static = False):
        for (a, s) in self.actors.copy():
            if static is None or s == static:
                self.actors.remove((a, s))

    def draw_setup(self, time = 0, static = None):
        if static is None or static:
            self.checked_draw_static()
        if static is None or not static:
            self.checked_draw(time)
        for item in self.items:
            item.draw_setup(time, static)

    def update(self, time, static = False):
        for item in self.items:
            item.update(time, static)
        self.remove_actors(static)
        if static is None or static:
            self.checked_draw_static()
        if static is None or not static:
            self.checked_draw(time)

class Scene(VtkBase, ResolveDefaults):
    _winsize = np.asarray([800, 600])
    _campos = np.asarray([0., 1., 7.])
    _camfocus = np.asarray([0., 0.5, 0.])
    _camup = np.asarray([0., 1., 0.])
    _camthick = None
    _camclip = None
    _name = "VTK Demo"
    _bg = np.asarray([0.8, 0.8, 0.8])
    _bg2 = np.asarray([0.2, 0.2, 0.2])
    _bggrad = True
    _lightfocus = np.asarray([0., 0., 0.])
    _lightpos = np.asarray([3., 5., 1.])
    _offscreen = False
    _backingstore = True

    def __init__(self, *args, **kwargs):
        self.actors = list()
        self.items = list()
        self.setups(*args, **kwargs)
        super().__init__(size = self.winsize)
        self.renderWindow.Render()

    def setups(self, *args, **kwargs):
        self.setup_parameters(*args, **kwargs)
        self.setup_renderer()
        time = kwargs.get('time', 0)
        self.build_model(time)
        self.draw(time)
        self.setup_background(time)
        self.setup_lighting(time)
        self.setup_camera(time)
        self.setup_renderwindow()
        self.setup_interactive()

    def setup_parameters(self, *args, **kwargs):
        self.resolve_defaults(kwargs)

    def build_model(self, time = None):
        pass

    def setup_renderer(self):
        # Create a renderer, render window, and interactor
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackingStore(self.backingstore)
        # renderer.UseShadowsOn()

    def setup_renderwindow(self):
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetOffScreenRendering(self.offscreen)
        self.renderWindow.SetSize(self.winsize)

        if self.name is not None:
            self.renderWindow.SetWindowName(self.name)
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindow.SetAlphaBitPlanes(1)

        # nvida pixel depth
        # self.renderWindow.SetAlphaBitPlanes(1)
        # self.renderWindow.SetMultiSamples(0)

        # self.renderWindow.Start() #No!

    def setup_interactive(self):
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindowInteractor.Initialize()

    def setup_camera(self, time):
        # camera
        camera = vtk.vtkCamera()
        # renderer.ResetCamera()
        if self.campos is not None:
            camera.SetPosition(self.campos)
        if self.camfocus is not None:
            camera.SetFocalPoint(self.camfocus)
        if self.camup is not None:
            camera.SetViewUp(self.camup)
        if self.camclip is not None:
            camera.SetClippingRange(self.camclip)
        if self.camthick is not None:
            if self.camthick < 0:
                thick = camera.GetThickness()
                self.camthick = max(thick, -self.camthick)
            camera.SetThickness(self.camthick)
            print(f'[DEBUG] Setting cam thickness to {self.camthick}')
        # add more properties
        self.camera = camera
        self.renderer.SetActiveCamera(camera)

    def setup_background(self, time):
        # set background
        if self.bg is not None:
            self.renderer.SetBackground(self.bg)
        if self.bg2 is not None:
            self.renderer.SetBackground2(self.bg2)
        if self.bggrad is not None:
            self.renderer.SetGradientBackground(self.bggrad)

    def setup_lighting(self, time):
        # light
        light = vtk.vtkLight()
        if self.lightfocus is not None:
            light.SetFocalPoint(self.lightfocus)
        if self.lightpos is not None:
            light.SetPosition(self.lightpos)
        self.renderer.AddLight(light)

    def clear(self):
        self.clear_actors()
        self.clear_items()

    def clear_items(self):
        self.items.clear()

    def clear_actors(self):
        for a in self.actors.copy():
            self.renderer.RemoveActor(a)
        self.actors.clear()

    def get_actors(self):
        actors = list()
        for item in self.items:
            actors.extend(item.get_actors())
        return actors

    def draw(self, time = 0):
        self.clear_actors()
        for item in self.items:
            item.draw_setup(time)
        self.actors.extend(self.get_actors())
        for a in self.actors:
            self.renderer.AddActor(a)

    def add(self, item):
        if isinstance(item, SceneItem):
            self.items.append(item)
        elif isinstance(item, (list, tuple)):
            for i in item:
                self.add(i)

    def update_items(self, time):
        pass

    def update(self, time = None):
        if time is None:
            return self
        old = list()
        for item in self.items:
            old.extend(item.get_items())
        old = np.asarray(old)
        self.update_items(time)
        new = list()
        for item in self.items:
            new.extend(item.get_items())
        new = np.asarray(new)
        ii = ~np.isin(new, old)
        for item in new[ii]:
            item.checked_draw_static()
        actors = list()
        for item in self.items:
            item.update(time)
            actors.extend(item.get_actors())
        new = np.asarray(actors)
        old = np.asarray(self.actors)
        ii = ~np.isin(old, new)
        for a in old[ii]:
            self.renderer.RemoveActor(a)
            self.actors.remove(a)
        ii = ~np.isin(new, old)
        for a in new[ii]:
            self.renderer.AddActor(a)
            self.actors.append(a)
        self.renderWindow.Render()
        return self

    def finalize(self):
        self.renderWindowInteractor.SetRenderWindow(None)
        self.renderWindowInteractor = None
        self.renderWindow.RemoveRenderer(self.renderer)
        self.renderWindow.Finalize()
        self.renderWindow = None

class AnimationBase(ResolveDefaults):
    _movie = None
    _delay = 1/60
    _format = None
    _transparent = False
    _magnification = None
    _antialias = False
    _processor = None
    _antialias = False
    _collect = False
    _parallel = True
    # TODO - add more antialias defaults, or allow dict
    # _logo = None
    _logo = (
        dict(
            logo = '~/Documents/Monash/MoCA/mocalogo.png',
            size = -.1,
            pos = (.02j, 0.98j),
            align = (0, 1j),
            ),
        dict(
            logo = '~/Documents/Monash/Logo/monashlogo.png',
            size = -.1,
            pos = (.02j, .92j),
            align = (0, 1j),
            ),
        )
    _cc_note = dict(
        text = "\u00A9 Alexander Heger (2019)",
        pos = (1j, 1j),
        align = (1j, 1j),
        size = 16,
        color = '#80808080',
        angle = 90,
        font = 'Roboto-Light.ttf',
        )
    _timestamp = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolve_defaults(kwargs)

        # create logos
        if self.logo is not None:
            if self.processor is None:
                self.processor = ProcessorChain()
            elif not isinstance(self.processor, ProcessorChain):
                self.processor = ProcessorChain(self.processor)
            if not isinstance(self.logo, (tuple, list, np.ndarray)):
                logo = self.logo,
            else:
                logo = self.logo
            for l in logo:
                self.processor.append(ImgLogoProcessor(**l))

        # create CC
        # if self.cc_note is not None:
        #     self.cc_note['text'] = kwargs.get('cc_note.text', self.cc_note['text'])
        if self.cc_note is not None and self.cc_note['text'] is not None:
            if self.processor is None:
                self.processor = ProcessorChain()
            elif not isinstance(self.processor, ProcessorChain):
                self.processor = ProcessorChain(self.processor)
            if not isinstance(self.cc_note, (tuple, list, np.ndarray)):
                cc = self.cc_note,
            else:
                cc = self.cc_note
            for c in cc:
                self.processor.append(FontProcessor(**c))

        # create timestamp(s)
        if self.timestamp is not None:
            if self.processor is None:
                self.processor = ProcessorChain()
            elif not isinstance(self.processor, ProcessorChain):
                self.processor = ProcessorChain(self.processor)
            if not isinstance(self.timestamp, (tuple, list, np.ndarray)):
                ts = self.timestamp,
            else:
                ts = self.timestamp
            for tsx in ts:
                tsx = tsx.copy()
                tsx['text'] = getattr(self, tsx['text'])
                self.processor.append(FontProcessor(**tsx))

        self.codec = kwargs.get('codec', None)

    def get_movie_writer(
            self,
            movie, batch = False,
            getter = None,
            ):
        kw = dict()
        if self.codec is not None:
            kw['codec'] = self.codec
        if self.parallel is not None:
            kw['parallel'] = self.parallel
        writer = MovieWriter(
            movie,
            batch = batch,
            delay = self.delay,
            format = self.format,
            collect = self.collect,
            getter = getter,
            processor = self.processor,
            getter_kw = dict(
                transparent = self.transparent,
                magnification = self.magnification,
                antialias = self.antialias,
                ),
            **kw
            )
        return writer

class Animation(AnimationBase):
    _nstep = 3600
    _time = None
    _scene = None
    _timescale = None
    _timeoffset = 0
    _timefunction = None
    _startframe = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess(args, kwargs)
        self.resolve_defaults(kwargs)
        self.scene = self.getscene(*args, **kwargs)
        if isinstance(self.scene, type):
            self.scene = self.scene(*args, **kwargs)
        if self.time is None:
            self.time = self.delay * self.nstep
        else:
            if self.delay is None:
                self.delay = self.time / self.nstep
            else:
                self.nstep = int(self.time / self.delay)
        if self.timescale is None:
            self.timescale = getattr(self.scene, 'timescale', 1.)
        self.init(*args, **kwargs)

    def preprocess(self, args, kwargs):
        pass

    def getscene(self, *args, **kwargs):
        scene = self.default(kwargs, 'scene')
        if scene is None:
            raise NotImplementedError()
        return scene

    def init(self, *args, **kwargs):
        pass

    def run(self, movie = None):
        scene = self.scene
        renderer = scene.renderer
        renderWindow = scene.renderWindow
        camera = scene.camera

        # add some settings storage
        camfocal0 = np.asarray(self.scene.camera.GetFocalPoint())
        camup0 = np.asarray(camera.GetViewUp())
        campos0 = np.asarray(camera.GetPosition())

        if self.transparent:
            bg0 = renderer.GetBackground()
            gbg0 = renderer.GetGradientBackground()
            renderer.SetBackground([1., 1., 1.])
            renderer.SetGradientBackground(False)

        if movie is None:
            movie = self.movie

        self.setup()
        if movie is not None:
            if isinstance(movie, BaseMovieWriter):
                writer = movie
            else:
                writer = self.get_movie_writer(movie)
            writer.set_getter(scene.get_frame)
        else:
            writer = NoMovieWriter()

        with writer.writing():
            # a movie
            t0 = time.time()
            for i in range(0, self.nstep):
                self.draw(i)
                renderWindow.Render()
                if self.movie is None:
                    dt = (t0 + self.delay) - time.time()
                    if dt > 0:
                        time.sleep(dt)
                    t0 = time.time()
                writer.write()

        self.cleanup()

        # restore settings
        if self.transparent:
            renderer.SetBackground(bg0)
            renderer.SetGradientBackground(gbg0)

        camera.SetFocalPoint(camfocal0)
        camera.SetPosition(campos0)
        camera.SetViewUp(camup0)
        self.scene.finalize()
        self.scene = None

    def setup(self, *args, **kwargs):
        pass

    def timefunc(self, iframe):
        """
        can overwrite time function by class or KW
        """
        if self.timefunction is not None:
            return self.timefunction(iframe)
        return self.timeoffset + self.delay * self.timescale * iframe

    def draw(self, iframe):
        time = self.timefunc(iframe)
        self.scene.update(time)

    def cleanup(self):
        pass


class Story(AnimationBase):
    """
    collection of animations to be played as a sequence

    Provide list of tuples with Animation and kwargs in `_animations` field.
    """
    # overwrite
    _animations = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolve_defaults(kwargs)
        self.kwargs = kwargs

    def run(self, movie = None):
        if movie is None:
            movie = self.movie
        if movie is None:
            writer = NoMovieWriter(batch = True)
        else:
            writer = MovieWriter(movie, batch = True)

        if self.animations is None:
            self.animations = tuple()

        for a in self.animations:
            if isinstance(a, (tuple, list)):
                a,k = a
            else:
                k = None
            if k is None:
                k = dict()
            k.update(self.kwargs)
            a(**k).run(movie = writer)

        writer.close()
