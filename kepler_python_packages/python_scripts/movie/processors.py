import os.path

import numpy as np

from PIL import Image, ImageFont, ImageDraw

########################################################################
# Example Processors
class BaseProcessor(object):
    def __init__(self):
        super().__init__()
    def __call__(self, frame):
        """
        return modified frame
        """
        return frame

class LogoProcessor(BaseProcessor):
    def __init__(self, pos, size, align = None, pos_mode = 'extended'):
        super().__init__()
        self.pos = np.asarray(pos)
        if size is not None:
            self.size = np.asarray(size)
        if align is None:
            align = np.array([0,0])
        self.align = align
        if pos_mode is None:
            pos_mode = 'extended'
        self.pos_mode = pos_mode
    def get_pos(self, img_size = None, size = None):
        if size is None:
           size = self.size
        if not hasattr(self, 'img_size'):
            self.img_size = np.array(img_size)
        if self.pos_mode == 'extended':
            pos = np.real(self.pos) + np.imag(self.pos) * self.img_size
        elif self.pos_mode == 'simple':
            # maybe this mode is more intuitive though less flexible?
            pos = np.real(self.pos) + np.imag(self.pos) * (self.img_size - size)
            ii = np.real(self.pos) < 0
            pos[ii] += self.img_size[ii] - size[ii]
        else:
            raise Exception(f'Unknown pos mode "{self.pos_mode}".')
        if self.align is not None:
            pos -= np.imag(self.align) * size + np.real(self.align)
        pos = np.asarray(np.round(np.real(pos)), dtype = np.int)
        pos = tuple(pos)
        return pos

class ImgLogoProcessor(LogoProcessor):
    def __init__(self, logo, pos, size = None, mask = None, **kwargs):
        self.pos = np.asarray(pos)
        if isinstance(logo, str):
            logo = os.path.expanduser(logo)
            logo = os.path.expandvars(logo)
            logo = Image.open(logo)
        if isinstance(logo, np.ndarray):
            logo = Image.fromarray(logo)
        if size is not None:
            if not np.shape(size) == (2,):
                if 0 < size < 8:
                    size = np.round(np.array(logo.size) * size)
                elif 0 > size > -8:
                    self.size = size
                elif size <= -8:
                    lsize = np.array(logo.size)
                    size = np.array(np.round(lsize * size / lsize[1]),
                                    dtype=np.uint)
                else:
                    lsize = np.array(logo.size)
                    size = np.array(np.round(lsize * size / lsize[0]),
                                    dtype=np.uint)
            if np.shape(size) == (2,):
                logo = logo.resize(size, resample=Image.LANCZOS)
        if mask is None:
            if logo.mode == 'RGBA':
                mask = logo
        elif np.shape(mask) == (3,):
            ii = np.array(logo) == mask
            mask = np.tile(255, ii.shape)
            mask[ii] = 0
            # alternatively
            # mask = LA.norm(np.array(logo) - mask[..., :]) / 255
            mask = Image.fromarray(mask, mode = 'L')
        self.logo = logo
        self.size = np.asarray(self.logo.size)
        self.mask = mask
        super().__init__(pos, size, **kwargs)
    def __call__(self, frame):
        img = Image.fromarray(frame.arr)
        if np.shape(self.size) != (2,):
            size = np.asarray(self.logo.size) * np.min(np.asarray(img.size) / np.asarray(self.logo.size)) * np.abs(self.size)
            size = np.array(np.round(size), dtype=np.uint)
            self.logo = self.logo.resize(size, resample=Image.LANCZOS)
            if self.mask is not None:
                self.mask = self.mask.resize(size, resample=Image.LANCZOS)
            self.size = self.logo.size
        pos = self.get_pos(img.size)
        img.paste(self.logo, box = pos, mask = self.mask)
        frame.set_arr(np.array(img))
        return frame

class FontProcessor(LogoProcessor):
    def __init__(self, text, pos,
                 size = 12,
                 color = 'black',
                 font = 'Arial.ttf',
                 angle = None,
                 **kwargs):
        self.text = text
        self.color = color
        self.font = ImageFont.truetype(font, size = size)
        self.angle = angle
        if self.angle is None:
            self.k = 0
        else:
            self.k = int(np.round(angle)) // 90
        if callable(self.text):
            size = None
        else:
            size = self.get_size(self.text)
        super().__init__(pos, size, **kwargs)

    def get_size(self, text):
        size = self.font.getsize(text)
        if self.k % 2 == 1:
            size = size[::-1]
        return size

    def __call__(self, frame):
        arr = frame.arr
        img_size = np.array(arr.shape[1::-1])
        if callable(self.text):
            text = self.text(frame.iframe)
            size = self.get_size(text)
        else:
            text = self.text
            size = None
        pos = self.get_pos(img_size, size = size)
        if self.k > 0:
            arr = np.rot90(arr, 4 - self.k)
            if self.k % 4 == 2:
                pos = img_size - pos - self.size
            if self.k % 4 == 1:
                pos = np.array(
                    [img_size[1] - pos[1] - self.size[1],
                     pos[0]])
            if self.k % 4 == 3:
                pos = np.array(
                    [pos[1],
                    img_size[0] - pos[0] - self.size[0],
                     ])
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.text(pos, text, self.color, self.font)
        arr = np.array(img)
        if self.k > 0:
            arr = np.rot90(arr, self.k)
        frame.set_arr(arr)
        return frame

class RotFontProcessor(LogoProcessor):
    def __init__(self, text, pos,
                 size = 12,
                 color = 'black',
                 font = 'Arial.ttf',
                 angle = None,
                 aa = 4,
                 **kwargs):
        font = ImageFont.truetype(font, size = size * aa)
        size = font.getsize(text)
        center = np.array((np.max(size),) * 2)
        center = ((center + (aa - 1)) // aa) * aa
        imsize = center * 2
        canvas = Image.new(
            mode = 'RGBA',
            size = tuple(imsize),
            color = (0, 0, 0, 0),
            )
        draw = ImageDraw.Draw(canvas)
        draw.text(center, text, color, font)
        canvas = canvas.rotate(angle)
        phi = np.pi * angle / 180
        c = np.cos(phi)
        s = np.sin(phi)
        rot = np.array([[c, s],[-s, c]])
        corners = np.array([[0, 0], size, [0, size[1]], [size[0], 0]])
        rec = center + np.tensordot(rot, corners, axes = (-1, -1)).transpose()
        rec = np.array([np.min(rec, axis = 0), np.max(rec, axis = 0)])
        rec = ((rec + (aa - 1)) // aa ) * aa
        rec = tuple(np.sort(rec, axis=0).flat)
        logo = canvas.crop(rec)
        logo = logo.resize(tuple(np.array(logo.size) // aa), resample=Image.HAMMING)
        size = logo.size
        self.logo = logo
        super().__init__(pos, size, **kwargs)
    def __call__(self, frame):
        img = Image.fromarray(frame.arr)
        pos = self.get_pos(img.size)
        img.paste(self.logo, box = pos, mask = self.logo)
        frame.set_arr(np.array(img))
        return frame

class BaseFinisher(BaseProcessor):
    pass

class ProcessorChain(BaseProcessor):
    def __init__(self, *processors):
        super().__init__()
        if processors == (None,):
            processors = tuple()
        self.processors = processors
    def __call__(self, frame):
        for processor in self.processors:
            frame = processor(frame)
        return frame
    def __setitem__(self, index, processor):
        self.processors = (
            self.processors[:index] +
            (processor,) +
            self.processors[index + 1:])
    def __delitem__(self, index):
        self.processors = (
            self.processors[:index] +
            self.processors[index + 1:])
    def pop(self):
        processor = self.processors[-1]
        self.processors = self.processors[:-1]
        return processor
    def append(self, processor):
        self.processors = self.processors + (processor,)
    def insert(self, index, processor):
        self.processors = (
            self.processors[:index] +
            (processor,) +
            self.processors[index:])
    def __len__(self):
        return len(self.processors)
