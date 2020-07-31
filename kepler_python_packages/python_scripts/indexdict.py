import collections.abc

class IndexDict(collections.abc.Mapping):
    """A dictonary that is also accessible by index."""

    def __init__(self,
                 keys,
                 values,
                 tolerant = False):
        """
        Key data can only be loaded on construction.

        Attibutes:
            keys -- list of keys, should be strings
            values -- list of values, should have same length

        Provides:
            access to vales of IndexDict x by:
                x[key]
                x[index]
                x(key)
                x(index)
                x.key
            get index of key:
                x.index(key)
            get key from index:
                x.key(index)

            You may also assign stuff, but type is not checked for now.
            For now, we just convert...

        ToDo:
            Slicing and sets of indices from list etc. would be nice.
            Some improved validation would be nice.
            Default value for get (default=None) instead of Error?
              (intensionally not, for now, for testing)
              (or, maybe better to allow user to decide what to do
               in case the key does not exist?)
            Type checking on assignment.
               Maybe set a flag whether to check?
               Maybe provide type information for conversion?
        """
        self.xkeys = {j:i for i,j in enumerate(keys)}
        self.list = list(keys)
        self.data = list(values)
        self.size = min(len(self.xkeys),len(self.data))
        self.tolerant = tolerant

    def _get_index(self,
                   index,
                   tolerant = False):
        """
        Return index from key or index.
        """
        try:
            i = int(index)
        except:
            i = self.xkeys.get(index, -1)
        if (i < 0) or (i >= self.size):
            if tolerant:
                return -1
            raise AttributeError("Key not found: {}".format(index))
        return i

    def get(self, *args):
        """
        similar to dict get
        """
        if len(args) == 0:
            raise TypeError(f"get() missing 1 required positional argument: 'key'")
        if len(args) > 2:
            raise TypeError(f"get() takes from 1 to 2 positional arguments but {len(args)} were given")
        try:
            return self._get(args[0])
        except:
            return args[1]

    def _get(self,
             index):
        """
        Return value by key or index.
        """
        i = self._get_index(index,
                           tolerant = self.tolerant)
        if i < 0 and self.tolerant:
            return None
        return self.data[i]

    def _set(self, index, value):
        """
        Assign value by key or index.
        """
        i = self._get_index(index)
        self.data[i] = type(self.data[i])(value)

    def __iter__(self):
        for k in self.xkeys:
            yield k

    def __contains__(self, key):
        return self.has_key(key)

    def __getitem__(self, index):
        """Interface to '[key/index]' indexing."""
        return self._get(index)

    def __setitem__(self, index, value):
        """Interface to '[key/index]' indexing."""
        return self._set(index, value)

    def __getattr__(self, attr):
        """
        Interface to '.key' indexing.
        """
        if attr not in self.__dict__:
            if attr in self.xkeys:
                return self._get(attr)
            raise AttributeError()
        super().__getattr__(attr)

    def has_key(self, key):
        try:
            i = self._get_index(key)
        except AttributeError:
            return False
        return self.data is not None

    def __setattr__(self, attr, value):
        if attr not in self.__dict__:
            if not attr in ('xkeys', 'data', 'size', 'list', 'tolerant'):
                if attr in self.xkeys:
                    self._set(attr, value)
                    return
                raise AttributeError("Key not found: {}".format(attr))
        super().__setattr__(attr, value)

    def __call__(self, attr):
        """Interface to "(key/index)" indexing."""
        return self._get(attr)

    def __len__(self):
        """Interface to len(<dictionary>)."""
        return self.size

    def index(self, key):
        """Return index of a given key."""
        i = self.xkeys.get(key, -1)
        if i < 0:
            raise AttributeError("Key not found: {}".format(key))
        return i

    def key(self, index):
        """
        Return key to a given index.
        """
        if index < 0 or index >= self.size:
            raise AttributeError("Index out of range: {}".format(index))
        return self.list[index]

    name = key
    number = index

    # def print(self, first = 1, last = None):
    #     """
    #     Print the parameters and values.
    #     """
    #     if last is None:
    #         last = self.size - 1
    #     for i in range(first, last + 1):
    #         print("{:>8} ({:>3d}): {!s}".format(self.key(i),i,self.data[i]))
