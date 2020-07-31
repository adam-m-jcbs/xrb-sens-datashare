from copy import copy

import numpy as np

from matplotlib.colors import to_rgba

from .frames import Frame


########################################################################
# Movie Managers

class BaseMovieManager(object):
    def __init__(self, start = 0):
        super().__init__()
        self.start = start
        self.reset()
    def __call__(self, frame):
        self.count += 1
        return (frame, )
    def reset(self):
        self.count = self.start
    def set_start(self, start, reset = True):
        self.start = start
        if reset:
            self.count = self.start
    def close(self):
        return tuple()
    # add special 'sync' signal chain?


class ChainMovieManager(BaseMovieManager):
    def __init__(self, *managers):
        super().__init__()
        if managers == (None,):
            managers = list()
        self.managers = managers

    def __call__(self, frame):
        outframes = [frame]
        for manager in self.managers:
            inframes, outframes = outframes, []
            for frame in inframes:
                outframes.extend(manager(frame))
        return tuple(outframes)

    def reset(self):
        for manager in self.managers:
            manager.reset()

    def set_start(self, start, reset = True):
        for manager in self.managers:
            manager.set_start(start, reset)

    def close(self):
        outframes = []
        for manager in self.managers:
            inframes, outframes = outframes, []
            for frame in inframes:
                outframes.extend(manager(frame))
            outframes.extend(manager.close())
        return tuple(outframes)

    # manage managers
    def __setitem__(self, index, manager):
        self.managers[index] = manager
    def __delitem__(self, index):
        del self.managers[index]
    def pop(self, *args):
        return self.managers.pop(*args)
    def append(self, manager):
        self.managers.append[manager]
    def insert(self, index, manager):
        self.managers.insert(index, manager)
    def __len__(self):
        return len(self.managers)


# Manager examples - logical managers

class ReplayMovieManager(BaseMovieManager):
    def __init__(self, *, copies = 2, **kwargs):
        super().__init__(**kwargs)
        self.copies = copies
        self.frames = list()

    def __call__(self, frame):
        self.frames.append(copy(frame))
        if self.copies <= 0:
            return tuple()
        self.count += 1
        return (frame, )

    def close(self):
        outframes = self.frames * (self.copies - 1)
        self.count += len(outframes)
        return tuple(outframes)

class CopyMovieManager(BaseMovieManager):
    def __init__(self, *, copies = 2, **kwargs):
        super().__init__(**kwargs)
        self.copies = copies

    def __call__(self, frame):
        if self.copies <= 0:
            return tuple()
        outframes = [frame] * self.copies
        self.count += len(outframes)
        return tuple(outframes)

class ReverseMovieManager(BaseMovieManager):
    def __init__(self, cycle = False, first = None, last = None, **kwargs):
        super().__init__(**kwargs)
        self.cycle = cycle
        assert cycle in (True, False)
        if cycle is True:
            if last is None:
                last = False
            if first is None:
                first = False
        else:
            if last is None:
                last = True
            if first is None:
                first = True
        self.first = first
        self.last = last
        self.frames = list()

    def __call__(self, frame):
        self.frames.append(copy(frame))
        if self.cycle:
            self.count += 1
            return (frame, )
        return tuple()

    def close(self):
        if not self.first:
            self.frames.pop(0)
        if not self.last:
            self.frames.pop(-1)
        self.count += len(self.frames)
        return tuple(self.frames[::-1])

class FrameReplicatorMovieManager(BaseMovieManager):
    """
    replicate select inframes

    provide dictionary of copies: {frameno: copies}, (no 0-based, default copies 1)
    """
    def __init__(self, copies = dict(), **kwargs):
        super().__init__(**kwargs)
        self.nbuffer = 0
        for k in copies.keys():
            if k < 0:
                self.nbuffer = max(self.nbuffer, -k)
        self.buffer = []
        self.copies = copies
        self.icount = -1
    def __call__(self, frame):
        if self.nbuffer > 0:
            self.buffer.append(copy(frame))
            if len(self.buffer) <= self.nbuffer:
                return tuple()
            frame = self.buffer.pop(0)
        self.icount += 1
        copies = self.copies.get(self.icount, 1)
        frames = [frame] * copies
        self.count += copies
        return tuple(frames)
    def close(self):
        frames = list()
        nframes = self.icount + len(self.buffer) + 1
        cframes = Counter()
        for k,n in self.copies.items():
            if k < 0:
                k += nframes
            cframes[k] += n
        for frame in self.buffer:
            self.icount += 1
            copies = cframes.get(self.icount, 1)
            frames += [frame] * copies
        self.count += len(self.frames)
        return tuple(frames)

class StoreLastFrameManager(BaseMovieManager):
    def __init__(self, **kwargs):
        super().__init__()
        self.frame = None

    def __call__(self, frame):
        self.frame = frame
        self.count += 1
        return (frame, )

    def getframe(self):
        return self.frame

# Manager examples - processing managers

class BlendInMovieManager(BaseMovieManager):
    def __init__(self, frame, nframes, func = None, **kwargs):
        super().__init__()
        self.nframes = nframes
        self.frame = frame
        if func is None:
            func = lambda x: 0.5 * (1 - cos(x * np.pi))
        self.func = func
        self.first = False

    def __call__(self, frame):
        if not self.first:
            return (frame, )
        self.first = False
        if isinstance(self.frame, str):
            col = np.astype(np.array(to_rgba(self.frame))*255, np.uint8)
            xframe = copy(frame)
            xframe[:, :, :] = col[np.newaxis, np.newaxis, :]
            self.frame = xframe
        values = np.arange(self.nframes) / (self.nframes-1)
        frames = []
        for x in values:
            f = self.func(x)
            if f == 0:
                xframe = self.frame
            elif f == 1:
                xframe = frame
            else:
                xframe = f * frame + (1 - f) * self.frame
                xframe =  np.astype(xframe,np.uint8)
            frames.append(xframe)
        return tuple(frames)

class BlendOutMovieManager(BaseMovieManager):
    def __init__(self, frame, nframes, func = None, **kwargs):
        super().__init__()
        self.nframes = nframes
        self.frame = frame
        if func is None:
            func = lambda x: 0.5 * (1 - cos(x * np.pi))
        self.func = func
        self.last = None

    def __call__(self, frame):
        self.last = frame
        return (frame, )

    def close(self):
        if isinstance(self.frame, str):
            col = np.astype(np.array(to_rgba(self.frame))*255, np.uint8)
            xframe = copy(self.last)
            xframe[:, :, :] = col[np.newaxis, np.newaxis, :]
            self.frame = xframe
        values = np.arange(self.nframes) / (self.nframes-1)
        frames = []
        for x in values:
            f = self.func(x)
            if f == 0:
                xframe = self.last
            elif f == 1:
                xframe = self.frame
            else:
                xframe = f * self.frame + (1 - f) * self.last
                xframe =  np.astype(xframe,np.uint8)
            frames.append(xframe)
        return tuple(frames)

class ApplyProcessorMovieManager(BaseMovieManager):
    """
    Use this to apply processor to subset of sequence

    provide range as slice
    currently step needs to be 1 or None
    """
    def __init__(self, processor, frames = slice(None), **kwargs):
        super().__init__()
        self.frames = frames
        self.processor = processor
        self.count = -1
        assert isinstance(frames, slice)
        assert frames.step in (None, 1)
        self.frames = frames
        store = 0
        if frames.start is not None and frames.start < 0:
            store = max(store, -frames.start)
        if frames.stop is not None and frames.stop < 0:
            store = max(store, -frames.stop)
        self.store = store
        start = 0
        if frame.start is not None and frame.start > 0:
            start = frame.start
        self.start = start
        stop = 2**63
        if frame.stop is not None and frame.stop > 0:
            stop = frame.start
        self.stop = stop
        self.buffer = []

    def __call__(self, frame):
        if self.store > 0:
            self.buffer.append(copy(frame))
            if len(self.buffer) > self.store:
                frame = self.buffer.pop(0)
            else:
                return tuple()
        self.count += 1
        frame = self._process(frame)
        return (frame, )

    def _process(self, frame):
        if self.count >= self.start and self.count < self.stop:
            frame = self.processor(frame)
        return frame

    def close(self):
        frames = []
        nframes = self.count + len(self.buffer) + 1
        if self.frames.start < 0:
            self.start += nframes
        if self.frames.stop < 0:
            self.stop += nframes
        for frame in self.buffer:
            self.count += 1
            frame = self._process(frame)
            frames.append(frame)
        return tuple(frames)

class BlendInOutProcessorMovieManager(BaseMovieManager):
    """
    blend in/out use of filter according to weight function for first/last nframes frames

    outoffset negative or None (default) will count relative to end
    inoffset negative will count relative to end, None = 0

    offsets are for 0 blending

    1     /---------\
    0 ___/           \___
        |            |
        inoffset     outoffset

    """
    def __init__(self, processor, inframes = 0, outframes = 0, inoffset = None, outoffset = None, func = None, **kwargs):
        super().__init__(**kwargs)
        self.inframes = inframes
        self.outframes = outframes
        self.nbuffer = 0
        if inoffset is None:
            inoffset = 0
        if inoffset < 0:
            self.infinal = True
            self.nbuffer = max(self.nbuffer, inoffset)
        else:
            self.infinal = False
        self.inoffset = inoffset

        if outoffset is None or outoffset < 0:
            self.outfinal = True
            if outoffset is None:
                outoffset = 0
            self.nbuffer = max(self.nbuffer, outframes + outoffset)
        else:
            self.outfinal = False
        self.outoffset = outoffset
        self.processor = procssor
        if func is None:
            func = lambda x: 0.5 * (1 - cos(x * np.pi))
        self.func = func
        self.buffer = list()
        self.icount = -1

    def _process(self, frame, x):
        f = self.func(x)
        if f == 0:
            return frame
        if f == 1:
            return self.processor(frame)
        frame = (1 - f) * frame + f * self.processor(frame)
        frame =  np.astype(xframe, np.uint8)
        return frame

    def __call__(self, frame):
        if self.nbuffer > 0:
            self.buffer.append(copy(frame))
            if len(self.buffer) <= self.nbuffer:
                return tuple()
            frame = self.buffer.pop(0)
        x = 1
        self.icount += 1
        if not self.outfinal:
            x *= (min(max(self.outoffset - self.icount, 0), self.outframes-1) /
                  (self.outframes-1))
        if not self.infinal:
            x *= (min(max(self.icount - self.inoffset, 0), self.inframes-1) /
                  (self.inframes-1))
        frame = self._process(frame, x)
        return (frame, )

    def close(self):
        if self.nbuffer == 0:
            return tuple()
        frames = []
        for j, frame in enumerate(self.buffer):
            self.icount += 1
            jcount = j - len(self.buffer)
            x = 1
            if not self.outfinal:
                x *= (min(max(self.outoffset - self.icount, 0), self.outframes-1) /
                      (self.outframes-1))
            else:
                x *= (min(max(self.outoffset - jcount, 0), self.outframes-1) /
                      (self.outframes-1))
            if not self.infinal:
                x *= (min(max(self.icount - self.inoffset, 0), self.inframes-1) /
                      (self.inframes-1))
            else:
                x *= (min(max(jcount - self.inoffset, 0), self.inframes-1) /
                      (self.inframes-1))
            frame = self._process(frame, x)
            frames.append(frame)

        self.count += len(self.frames)
        return tuple(frames)

class BlendProcessorSequenceMovieManager(BaseMovieManager):
    def __init__(self, sequences = dict(), nframes = 0, inframes = 0, blendframes = 0, outframes = 0, func = None, **kwargs):
        super().__init__(**kwargs)
        self.inframes = inframes
        self.outframes = outframes
        self.blendframes = blendframes
        self.sequence = sequence
        self.nframes = nframes
        self.offset = offset
        if func is None:
            func = lambda x: 0.5 * (1 - cos(x * np.pi))
        if blendfunc is None:
            self.blendfunc = lambda x, y : (func(x), func(1-x))
        self.func = func
        self.buffer = list()
        self.nbuffer = 0
        for k in sequences.keys():
            if k < 0:
                self.nbuffer = max(self.nbuffer, -k)
        self.icount = -1

    def _process(self, frame, x, processor):
        f = self.func(x)
        if f == 0:
            return frame
        if f == 1:
            return processor(frame)
        frame = (1 - f) * frame + f * processor(frame)
        frame =  np.astype(xframe, np.uint8)
        return frame

    def _blendprocess(self, frame, x, processor0, processor1):
        f1, f2 = self.blendfunc(x)
        f = min(max(1 - f1 - f2, 0), 1)
        if f == 1:
            return frame
        if f1 == 1:
            return processor1(frame)
        if f2 == 1:
            return processor2(frame)
        frame = f * frame + f1 * processor1(frame) + f2 * processor2(frame)
        frame =  np.astype(xframe, np.uint8)
        return frame

    def _run_sequence(self, frame, seq):
        frames = list()
        if seq is None:
            frames.append(frame)
            return frames
        nseq = len(seq)
        flt0 = None
        for iseq, flt in enumerate(seq):
            if iseq == 0:
                inframes = self.inframes
                if inframes > 0:
                    values = np.arange(inframes) / inframes
                    for x in values:
                        frames.append(self._process(frame, x, flt))
            else:
                blendframes = self.blendframes
                if blendframes > 0:
                    values = np.arange(blendframes) / blendframes
                    for x in values:
                        frames.append(self._blendprocess(frame, x, flt0, flt))
            for _ in self.nframes:
                frames.append(self._process(frame, 1, flt))
            if iseq == nseq - 1:
                outframes = self.outframes
                if outframes > 0:
                    values = (outframes - np.arange(outframes) - 1) / outframes
                    for x in values:
                        frames.append(self._process(frame, x, flt))
            flt0 = flt
        return tuple(frames)

    def __call__(self, frame):
        if self.nbuffer > 0:
            self.buffer.append(copy(frame))
            if len(self.buffer) <= self.nbuffer:
                return tuple()
            frame = self.buffer.pop(0)
        self.icount += 1
        seq = self.seqeunces.get(self.icount, None)
        if seq is not None:
            frames = self._run_sequence(frame, seq)
        else:
            frames = (frame, )
        return tuple(frames)

    def close(self):
        if self.nbuffer == 0:
            return tuple()
        frames = list()
        for i,frame in enumerate(self.buffer):
            jcount = i - len(self.buffer)
            self.icount += 1
            seqi = self.seqeunces.get(self.icount, None)
            if seqi is not None:
                frames.extend(self._run_sequence(frame, seqi))
            seqj = self.seqeunces.get(jcount, None)
            if seqj is not None:
                frames.extend(self._run_sequence(frame, seqj))
            if seqi is None and seqj is None:
                frames.append(frame)
        self.count += len(self.frames)
        return tuple(frames)

class SortMovieManager(BaseMovieManager):
    BATCH_FINISH = object()
    def __init__(self, start=0, **kwargs):
        super().__init__()
        self.start = start
        self.nframe = start
        self.store = dict()

    def __call__(self, frame):
        iframe = frame.iframe
        if iframe is None:
            if len(self.store) == 0:
                iframe = self.nframe + 1
            else:
                iframe = max(self.store.keys()) + 1
        assert isinstance(frame, Frame)
        assert iframe > self.nframe
        assert iframe not in self.store
        self.store[iframe] = frame
        print(f'[{self.__class__.__name__}] Achiving Frame {iframe} ({len(self.store)}).')
        frames = list()
        while True:
            jframe = self.nframe + 1
            if jframe in self.store:
                frames.append((self.store.pop(jframe), jframe,))
                print(f'[{self.__class__.__name__}] Sending Frame {jframe} ({len(self.store)}).')
                nframe = jframe
            else:
                break
        return tuple(frames)

    def close(self):
        assert len(self.store) == 0
        return super().close()




# TODO - for example, to have sequence of text descriptions, apply sequence
# of processors blended in and out onto a single frame.  Or running sequence?

# A version to run on top of running movie to have font animation
# while movie is running.
# Maybe these would be a metafilter rather than a manager.

# TODO - filters for animated fonts
