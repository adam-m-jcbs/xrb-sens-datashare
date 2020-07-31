#TEST

class Property(object):
    """
    Emulate PyProperty_Type() in Objects/descrobject.c

    I changed this example to implement a write-once property
    """

    __slots__ = ('fget','fset','fdel','doc','iter')

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc
        self.iter=0

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.iter += 1
        if self.iter <= 1:
            self.fset(obj, value)
        else:
            print('Too much access',self.iter)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)
        self.iter = 0

class Strange(object):
    def fget(self):
        """We could define an init method insead..."""
        if not hasattr(self,'_x'):
            print('good try')
            return None
        print('getting')
        return self._x
    def fset(self,v):
        print('setting')
        self._x = v
#    x = Property(fget,fset)
    x = property(fget,fset,None,"A test.")


# In [50]: x=np.frompyfunc(isotope.Ion,2,1)

# In [51]: x(np.array(12),np.array(6))
# Out[51]: Ion('Mg6')

# In [52]: x(np.array([6,7,8]),np.array([0,0,0]))
# Out[52]: array([C, N, O], dtype=object)

# In [53]: x(np.array([6,7,8]),np.array([12,14,16]))
# Out[53]: array([C12, N14, O16], dtype=object)

def linkplus(link):
    val = 0
    for b in bytearray(link):
        val *= 51
        # g is not allowd
        # z is not allowed
        # a should be 1 not 0 other wise a = aa, ab = b, ...
        if 96 < b < 103:
            val += b - 96
        elif 103 < b < 122:
            val += b - 97
        elif 64 < b < 91:
            val += b - 40
        else:
            print("linkplus error with '{}'".format(link))
    print("val = {}".format(val))

    print(val)
    val += 1
    i = 1
    while val >= 51**i:
        if divmod(val,51**i)[1] < 51**(i-1):
            val += 51**(i-1)
        i += 1
    link = ''
    while val > 0:
        val, c = divmod(val, 51)
        if c > 24:
            link = chr(c + 40) + link
        elif c < 7:
            link = chr(c + 96) + link
        else:
            link = chr(c + 97) + link
    return link


# # the following does not really have any use yet
# class AbuIter(object):
#     def __init__(self, abu):
#         self.abu = abu
#     def __iter__(self):
#         self.location = 0
#         return (self)
#     def next(self):
#         if self.location >= len(self.abu):
#             raise StopIteration
#         i = self.location
#         self.location = i+1
#         return (self.abu.iso[i].Name(),self.abu.abu[i])


#--- begin method 2
    # def listit(self):
    #     i=0
    #     while i < self.__len__():
    #         yield self.iso[i]
    #         i+=1
    # def __iter__(self):
    #        return self.listit()
#--- end method 2
#--- begin method 3
    # def __iter__(self):
    #     self.location = 0
    #     return self
    # def next(self):
    #     if self.location >= self.__len__():
    #         del self.location
    #         raise StopIteration
    #     i = self.location
    #     self.location = i+1
    #     return (self.iso[i],self.abu[i])
#--- end method 3
    # def iter_el(self):
    #      i = 0
    #      el = self.get_el()
    #      n = len(el)
    #      while i < n:
    #          yield el[i].Name()
    #          i += 1


def test1():
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from matplotlib.transforms import Bbox
    from matplotlib.axes import Axes

    x = np.array([[.1,.1],[.9,.9]])
    b = Bbox(x)
    f = plt.figure()
    a = Axes(f,b)
#    a = f.add_subplot(111,b)
    a._originalPosition = a._position
    f.add_axes(a)
    plt.draw()
    return f,a,b,x


# test class hirachy
class testA(object):
    def a(self):
        print('A.a')
        self.b()
    def b(self):
        print('A.b')

class testB(object):
    def a(self):
        print('B.a')
    def b(self):
        print('B.b')
        print((type(self)))

class testC(testA, testB):
    def x(self):
        self.a()
    def b(self):
        print('C.b')
        super(testC, self).b()

# test wrapper with parameter
# method 1 - function
def nfun(n):
    def wrap(fun):
        def wrapped(*args, **kwargs):
            x = []
            for i in range(n):
                x += [fun(*args,**kwargs)]
            return x
        return wrapped
    return wrap
@nfun(4)
def testfun(i):
    return i

# test wrapper with parameter
# method 2 - class
class nfunc(object):
    def __init__(self, n):
        self._n = n
    def __call__(self, fun):
        def wrapped(*args, **kwargs):
            x = []
            for i in range(self._n):
                x += [fun(*args, **kwargs)]
            return x
        return wrapped
@nfunc(5)
def testfunc(i):
    return i


# setting properties
import functools
from types import MethodType
def attrib(obj, name, func, doc = None):
    class F(object):
        def __init__(self, func):
            self.func = func
        def __get__(self, obj, objtype):
            print('getting')
            if obj is None:
                return self
            return self.func
        def __set__(self, obj, val):
            raise AttributeError()
        __doc__  = doc
    setattr(obj, name, F(func))

from utils import make_cached_attribute
import numpy as np
class T(object):
    def __init__(self, x):
        name = 'f'
        doc = "f-{:d}".format(x)
        cls = self.__class__
        self.z = np.arange(2*x**2).reshape((x,-1))
        def funx(self, idx):
            return (self.z[:,idx])
        make_cached_attribute(self, funx, name, doc, args=(x,))

# bad
# class T(object):
#     def __init__(self, x):
#         name = 'f'
#         doc = "f-{:d}".format(x)
#         cls = self.__class__
#         def func(self):
#             print(x)
#         func.__doc__  = doc
#         func.im_class = cls
#         func.im_func  = func
#         func.im_self  = None
#         setattr(cls, name, func)

from utils import CachedAttribute
class U(object):
    def __init__(self, x):
        self.x = x
        def f(self, obj, objtype = None):
            return 2
        self.y = property(f)
    @CachedAttribute
    def X(self):
        return self.x

# the following now is a singleton metaclass
# the commented singleton method on class basis is more clunky
#    because it requires the _init attribute to avoid re-initialization
class SMeta(type):
    def __init__(*args):
        print('type init', args)
        type.__init__(*args)
        # args[0].x = 3

    def __call__(*args):
        print(('type call', args ))
        cls = args[0]
        key = args[1:]
        try:
            cache = cls._cache
        except:
            cache = dict()
            cls._cache = cache
        try:
            obj = cache[key]
        except:
            obj = type.__call__(*args)
            cache[key] = obj
        # obj = type.__call__(*args)
        print('type call --> ', obj)
        return obj

    def __new__(*args):
        print('type new', args)
        obj = type.__new__(*args)
        print('type new --> ', obj)
        return obj

    def __getattribute__(*args):
        print("type getattribute invoked", args)
        return type.__getattribute__(*args)

class S(object, metaclass=SMeta):
    def __init__(*args):
        print("class init", args)
        # self = args[0]
        # if hasattr(self, '_init'):
        #     print('class init: NOT NEW')
        #     return
        # self._init = True
        object.__init__(*args)
    def __new__(*args):
        print("class new", args)
        # cls = args[0]
        # try:
        #     cache = cls._cache
        # except:
        #     cache = dict()
        #     cls._cache = cache
        # obj = cache.get(args[1:], None)
        # if obj is None:
        #     obj = object.__new__(*args)
        #     cache[args[1:]] = obj
        obj = object.__new__(*args)
        print("class new --> ",obj)
        return obj
    def __len__(self):
        return 10
    def __getattribute__(*args):
        print("class getattribute invoked", args)
        return object.__getattribute__(*args)


# the following now is a metaclass that creates subclasses instead
# is this possible? -- YES
# this allows for init to add specific methods (can only be on class level)
# dynamically - unfortunately there is no instance-level methods.
# I suppose these could be added in the instance's __new__ method?
class MMeta(type):
    """
    metaclass that creates subclasses instance instead of class
    instance itself when called
    """
    def __init__(*args):
        print('type init', args)
        type.__init__(*args)

    def __call__(*args):
        """
        return object with new subclass '_'+cls.__name__
        for each call (unless class name starts with '_'.
        """
        print('type call', args )
        cls = args[0]
        if cls.__name__.startswith('_'):
            obj = type.__call__(*args)
        else:
            prm = args[1:]
            MX = cls.__class__(
                '_' + cls.__name__,
                (cls,),
                dict())
            MX.__doc__ = cls.__doc__
            # the following would create an 'analog' class instead
            # that is not a subclass of cls
            # which seems less usefull in most cases
            # MX = cls.__class__(
            #     '_' + cls.__name__,
            #     cls.__bases__,
            #     dict(cls.__dict__))
            obj = MX(prm)
        print(('type call --> ', obj))
        return obj

    def __new__(*args):
        print('type new', args)
        obj = type.__new__(*args)
        print('type new --> ', obj)
        return obj

    # maybe this only works in python 3.x
    @classmethod
    def __prepare__(*args, **kwargs):
        print('type prepare', args, kwargs)
        obj = type.__prepare__(*args, **kwargs)
        print('type prepare --> ', obj)
        return obj

class M(object, metaclass = MMeta):
    __doc__ = """
    testclass that adds attributes to class definition in on
    initialization
    """
    def __init__(*args):
        """
        write arguments as 'data' into class
        """
        print("class init", args)
        object.__init__(*args)
        self = args[0]
        self.__class__.data = args[1:]
    def __new__(*args):
        print("class new", args)
        obj = object.__new__(*args)
        print(("class new --> ",obj))
        return obj
    def __len__(self):
        return 10
