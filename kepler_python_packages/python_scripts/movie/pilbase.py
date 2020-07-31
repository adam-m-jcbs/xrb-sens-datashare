"""
Some ideas on how to combin CAIRO graphics with Movie framework
"""

import numpy as np
import PIL

from PIL import Image
# from io import BytesIO

from .frames import MovieCanvasBase

class PilBase(MovieCanvasBase):
    def __init__(self,
                 size = (800, 600),
                 mode = 'RGBA',
                 color = (0, 0, 0, 0),
                 ):
        self.mode = mode
        self.size = size
        self.color = color
        self.image = Image.new(
            mode = mode,
            size = tuple(size),
            color = color,
            )

    def get_array(self):
        return np.array(self.image)

    def get_frame_size(self):
        return self.image.size

    @classmethod
    def from_arr(cls, arr):
        return Image.fromarr(arr, mode = 'RGBA')

    def get_image(self):
        return self.image.copy()

    def write_image(self, filename = None, format = None, **kwargs):
        self.image.save(filename, format = format, **kwargs)

    def close(self):
        self.image.close()
        del self.image

    def clear(self, color = None):
        if color is not None:
            color = self.color
        if color is None:
            color = (0, 0, 0, 0)
        if isinstance(color, tuple, np.ndarray):
            assert len(color) == 4
            color = np.array(color) * 2**(8 * np.array([0,1,2,3]))
        self.image.paste(col, (0, 0) + self.image.size)


import os.path
import re
from tempfile import TemporaryDirectory, NamedTemporaryFile
import sys
import subprocess
from pdf2image import convert_from_bytes
import time
import shutil

class TikZBase(PilBase):
    def __init__(
            self,
            template,
            dpi = 200,
            transparent = False,
            format = 'png',
            single_file = True,
            size = None,
            load_extra_pages = False,
            silent = False,
            ):

        self.dpi = dpi
        self.transparent = transparent
        self.format = format
        self.single_file = single_file
        self.size = size
        self.load_extra_pages = load_extra_pages
        self.silent = silent

        if len(template) < 256:
            filename = template
            filename = os.path.expandvars(filename)
            filename = os.path.expanduser(filename)
            self.filename = filename
            with open(filename, 'rt') as f:
                self.template = f.read()
        else:
            self.template = template
        self.clear()

    def clear(self):
        self.imgtime = 0
        self.pdftime = 0
        self.textime = 0
        if hasattr(self, 'image'):
            if self.image is not None:
                self.image.close()
        self.extra_images = []
        self.pdfdata = None
        self.texdata = self.template

    def close(self):
        self.clear()

    def _compile_tex(self, texdir, texfile):
        latex = shutil.which('pdflatex')
        cmd = [latex, '-shell-escape']
        basename = texfile.rsplit('.', 1)[0]
        args = cmd
        if self.silent and False:
            args += ['-interaction', 'batchmode']
        else:
            args += ['-interaction', 'nonstopmode']
        args += [basename]
        runs = 0
        logs = b''
        while True:
            runs += 1
            log = b''
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=texdir,
                )
            while proc.poll() is None:
                text = proc.stdout.readline()
                log += text
                if not self.silent:
                    sys.stdout.buffer.write(text)
            logs += log
            if re.search(
                    r'\(rerunfilecheck\)',
                    log.decode('us-ascii', 'ignore')
                    ) is not None:
                continue
            break
        self._tex_log = logs.decode('us-ascii', 'ignore')

    def _make_pdf(self):
        if self.textime < self.pdftime:
            return
        with TemporaryDirectory() as texdir:
            with NamedTemporaryFile('wt', suffix='.tex', dir=texdir) as texfile:
                texfile.write(self.texdata)
                self._compile_tex(texdir, texfile.name)
                pdffilename=texfile.name[:-3] + 'pdf'
            time.sleep(0.1)
            if os.path.isfile(pdffilename):
                with open(pdffilename, 'rb') as pdffile:
                    pdfdata = pdffile.read()
            else:
                print(self._tex_log)
                pdfdata = None
        self.pdfdata = pdfdata
        self.pdftime = time.time()

    def _make_image(self):
        self._make_pdf()
        if self.pdftime >= self.imgtime:
            if self.pdfdata is None:
                if self. size is None:
                    self.image = None
                else:
                    self.image = Image.new(
                        size = self.size,
                        color = (255,255,255,255),
                        mode = 'RGBA',
                        )
            else:
                self.image = convert_from_bytes(
                    self.pdfdata,
                    dpi = self.dpi,
                    transparent = self.transparent,
                    fmt = self.format,
                    single_file = self.single_file,
                    size = self.size)
                if (len(self.image) > 1) and self.extra_pages:
                    self.extra_pages = self.image[1:]
                    for i,p in enumerate(self.extra_pages):
                        p.load()
                        self.extra_pages[i] = p.convert('RGBA')
                else:
                    self.extra_pages = []
                self.image = self.image[0]
                self.image.load()
                self.image = self.image.convert('RGBA')
                if self.size is None:
                    self.size = self.image.size
            self.imgtime = time.time()

    def _apply_parameters(self, parameters=dict()):
        frame = self.texdata
        for pattern, repl in parameters.items():
            frame = re.sub(pattern, repl, frame, flags=re.MULTILINE)
        self.texdata = frame

    def draw(self, parameters = None):
        if parameters is not None and len(parameters) > 0:
            self._apply_parameters(parameters)
            self.textime = time.time()

    def get_pdf(self, filename = None):
        self._make_pdf()
        if filename is None:
            return self.pdfdata
        filename = os.path.expandvars(filename)
        filename = os.path.expanduser(filename)
        with open(filename, 'wb') as f:
            f.write(self.pdfdata)

    def get_image(self):
        self._make_image()
        return self.image.copy()

    def get_array(self):
        self._make_image()
        return np.array(self.image)

    def get_frame_size(self):
        if self.size is None:
            self._make_image()
        return self.size

    def write_image(self, *args, **kwargs):
        self._make_image()
        super().write_image(*args, **kwargs)

    def get_canvas(self):
        return TikZCanvas(self)

    def update_template_from_current(self):
        self.template = self.texdata

    def _setparm_bool(self, name, value):
        replace = dict()
        assert isinstance(value, bool)
        if value == True:
            replace[fr"^(?:\s|%)*(\\setboolean\{{{name}\}}\{{false\}})"] = fr"%\g<1>"
            replace[fr"^(?:\s|%)*(\\setboolean\{{{name}\}}\{{true\}})"] = fr"\g<1>"
        else:
            replace[fr"^(?:\s|%)*(\\setboolean\{{{name}\}}\{{true\}})"] = fr"%\g<1>"
            replace[fr"^(?:\s|%)*(\\setboolean\{{{name}\}}\{{false\}})"] = fr"\g<1>"
        return replace

    def _setparm_value(self, name, value):
        replace = dict()
        replace[fr"^\s*(\\newcommand\{{\\{name}\}}\{{)[^\}}]+(\}})"] = fr"\g<1>{value}\2"
        return replace

    def setparm(self, **values):
        replace = dict()
        for name, value in values.items():
            if isinstance(value, bool):
                replace.update(self._setparm_bool(name, value))
            else:
                replace.update(self._setparm_value(name, value))
        self.draw(replace)

class TikZCanvas(object):
    def __init__(self, base):
        self.base = base
    def __setattr__(self, attr, value):
        if (attr.startswith("_") or
                attr in ('base',) or
                not hasattr(self, 'base')
                ):
            return super().__setattr__(attr, value)
        self.base.setparm(**{attr: value})
    def __setitem__(self, key, value):
        self.base.setparm(**{key: value})
    # add more functionallity

from movie import ParallelMovie, PoolMovie, ReverseMovieManager
from numpy import mgrid, sqrt, cos, pi

class Cake():
    def __init__(
            self,
            fig = None,
            twdw = 120,
            pw = 2.3,
            tjdj = 120,
            pj = 2.3,
            fromw = True,
            angres = 1,
            template = '/home/alex/LaTeX/oblique/vectors.tex',
            ):

        if fig is None:
            fig = TikZBase(template).get_canvas()

        parameters = ('twdw', 'pw', 'tjdj', 'pj', 'fromw', 'angres')
        for p in parameters:
            fig[p] = locals()[p]

        self.fig = fig

    @classmethod
    def movie(
            cls,
            filename = '~/Downloads/cakew2_3.webm',
            ns = 720,
            template = '/home/alex/LaTeX/oblique/vectors3.tex',
            dpi = 200,
            pw = 1.7,
            angres = 1,
            pj = 2.3,
            fromw = True,
            nparallel = None,
            ):
        if fromw:
            key = 'twdw'
        else:
            key = 'tjdj'
        ParallelMovie.make(
        #PoolMovie.make(
            filename,
            func = cls,
            fkwargs = dict(
                angres=angres,
                pw=pw,
                pj=pj,
                fromw=fromw,
                ),
            values = mgrid[0:360:(ns+1)*1j][:-1] + 0.5 * 360 / ns,
            key = key,
            canvas = TikZBase,
            ckwargs = dict(
                template=template,
                silent=True,
                dpi = dpi,
                ),
            nparallel = nparallel,
            )

    @classmethod
    def movie2(
            cls,
            filename = '~/Downloads/cakes180.webm',
            ns = 720,
            template = '/home/alex/LaTeX/oblique/vectors3.tex',
            dpi = 200,
            lim = 10,
            twdw = 180,
            tjdj = 180,
            skew = 0.01,
            skep = 0,
            fromw = False,
            angres = 1,
            nparallel = None,
            ):
        if fromw:
            data = 'pw'
        else:
            data = 'pj'
        ParallelMovie.make(
        #PoolMovie.make(
            filename,
            func = cls,
            fkwargs = dict(
                angres=angres,
                twdw=twdw,
                tjdj=tjdj,
                fromw=fromw,
                ),
            values = 2 * lim**(cos(mgrid[0:pi:(ns//2+1)*1j]+skep)+skew),
            data = data,
            canvas = TikZBase,
            ckwargs = dict(
                template=template,
                silent=True,
                dpi = dpi,
                ),
            manager = ReverseMovieManager(cycle=True),
            nparallel = nparallel,
            )

    @classmethod
    def movie3(
            cls,
            filename = '~/Downloads/cakes0.webm',
            ns = 72,
            template = '/home/alex/LaTeX/oblique/vectors3.tex',
            dpi = 200,
            lim = 10,
            twdw = 0,
            tjdj = 0,
            fromw = False,
            angres = 1,
            nparallel = None,
            ):
        if fromw:
            data = 'pw'
        else:
            data = 'pj'
        ParallelMovie.make(
        #PoolMovie.make(
            filename,
            func = cls,
            fkwargs = dict(
                angres=angres,
                twdw=twdw,
                tjdj=tjdj,
                fromw=fromw,
                ),
            values = 2 * lim**(cos(mgrid[0:pi*(ns//2)/(ns//2+1):(ns//2)*1j]+0.5*pi/(ns//2+1))),
            data = data,
            canvas = TikZBase,
            ckwargs = dict(
                template=template,
                silent=True,
                dpi = dpi,
                ),
            manager = ReverseMovieManager(cycle=True, first=True, last=True),
            nparallel = nparallel,
            )


"""
seems to be possible to use inkscape to convert *svg* in pipe mode
(fails on other formats as if expects svg but does not recognise pdf)

cat $HOME/Documents/Monash/Logo/monashlogo.svg | inkscape -D -z --file=/dev/stdin --export-png=- -d 300 > yyy.png
"""


"""
\newcommand{\angw}{160}
\newcommand{\pw}{1.82}
\setboolean{mu}{false}
%\setboolean{mu}{true}

"""
