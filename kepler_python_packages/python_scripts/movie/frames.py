"""
Deal with frames
"""

import time
import functools
from abc import ABC, abstractmethod
from PIL import Image

# defind a sequence class
@functools.total_ordering
class Sequence(object):
    def __init__(self, seqid = None):
        if seqid is None:
            self.new()
        else:
            self.seqid = seqid

    def new(self):
        self.seqid = int(time.monotonic()*1e9)

    def copy(self):
        return self.__class__(self.seqid)

    def __hash__(self):
        return self.seqid

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return self.seqid == other.seqid

    def __lt__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return self.seqid < other.seqid

    @classmethod
    def none(cls):
        return cls(0)

# set up default sequence
def newsequence():
    global sequence, seqid
    sequence = Sequence()
    seqid = sequence.seqid
newsequence()


# define Frame base class
class Frame(object):
    def __init__(
            self,
            arr = None,
            iframe = None,
            oframe = None,
            seqid = None,
            seq = None,
            kind = None,
            ):

        self._arr = arr
        self._iframe = iframe
        self._oframe = oframe
        if seqid is not None and seq is None:
            seq = Sequence(seqid)
        if seq is None:
            seq = sequence
        self._seq = seq.copy()

    # could add flags to indicate last frame of a sequence.
    # though that may be tricky with manangers

    @property
    def seq(self):
        return self._seq

    @seq.setter
    def seq(self, seq):
        assert isinstance(seq, Sequence)
        self.seq = seq.copy()

    @property
    def iframe(self):
        return self._iframe

    @iframe.setter
    def iframe(self, iframe):
        self._iframe = iframe

    @property
    def oframe(self):
        return self._oframe

    @oframe.setter
    def oframe(self, oframe):
        self._oframe = oframe

    @property
    def seqid(self):
        """
        return frame, allow for lazy evaluation?
        """
        return self.seq.seqid

    @seqid.setter
    def seqid(self, seqid):
        self.seq = Sequence(seqid)

    @property
    def arr(self):
        """
        return frame, allow for lazy evaluation?
        """
        return self._arr

    @arr.setter
    def arr(self, arr):
        self._arr = arr


class MovieCanvasBase(ABC):
    def get_frame(self, iframe = None):
        """Interface function to get Frame object"""
        return Frame(self.get_array(), iframe = iframe)

    def get_buf_shape(self):
        """Helper function to get shape of buffer."""
        return self.get_frame_size() + (4,)

    def get_empty_frame(self, col = '#ffffffff', iframe = None, seq = None):
        arr = np.ndarray(self.get_buf_shape(), dtype=unit8)
        assert col[0] == '#'
        col = np.array([col[i,i+2] for i in range(1,8,2)])
        arr[:,:,:4] = col
        return Frame(arr, iframe = iframe, seq = seq)

    def get_image(self):
        """return PIL impage"""
        return Image.fromarray(self.get_array())

    def close(self):
        """Free resources when finished"""
        pass

    def clear(self):
        """reset canvas for next frame"""
        raise NotImplementedError()

    def get_canvas(self):
        """return object needed for rendering of content"""
        raise NotImplementedError()

    def write_image(self, filename = None):
        """write image to file"""
        raise NotImplementedError()

    @abstractmethod
    def get_array(self):
        """return raw data array - of shape (height, width, 4==depth)"""
        raise NotImplementedError()

    @abstractmethod
    def get_frame_size(self):
        """return size of frame"""
        raise NotImplementedError()
