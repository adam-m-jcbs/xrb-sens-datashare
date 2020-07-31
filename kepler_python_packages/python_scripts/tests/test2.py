# test2

class Meta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        print('Meta.__prepare__', metacls, name, bases, kwds)
        result = type.__prepare__(metacls, name, bases, **kwds)
        print('Meta.__prepare__', ' ==> ', result)
        return result

    def __new__(metacls, name, bases, namespace, **kwds):
        print('Meta.__new__', metacls, name, bases, namespace, kwds)
        result =  type.__new__(metacls, name, bases, namespace)
        print('Meta.__new__', ' ==> ', result)
        return result

    def __init__(cls, name, bases, namespace, **kwargs):
        print('Meta.__init__', cls, name, bases, namespace, kwargs)
        result = type.__init__(cls, name, bases, namespace)

        if kwargs.get('kw', True):
            x = cls.__call__
            def call(self, *args, **kwargs):
                r = x(self, *args, **kwargs)
                if kwargs.get('final', True):
                    print('applying transform')
                    r = (r,)
                return r
            cls.__call__ = call

        print('Meta.__init__', ' ==> ', result)
        return result

    def __call__(cls, *args, **kwargs):
        print('Meta.__call__', cls, args, kwargs)
        result =  type.__call__(cls, *args, **kwargs)
        print('Meta.__call__', ' ==> ', result)
        return result

class Class(object, metaclass=Meta, kw=True):
#class Class(object, metaclass=Meta):
#class Class(object):
    def __new__(cls, *args, **kwargs):
        print('Class.__new__', cls, args, kwargs)
        result =  super().__new__(cls)
        print('Class.__new__', ' ==> ', result)
        return result

    def __init__(self, *args, **kwargs):
        print('Class.__init__', self, args, kwargs)
        result =  super().__init__()
        print('Class.__init__', ' ==> ', result)
        return result

    def __call__(self, *args, **kwargs):
        print('Class.__call__', self, args, kwargs)
        result = (args, kwargs)
        print('Class.__call__', ' ==> ', result)
        return result


print('=== Done class definitions ===')

instance = Class('x', y='Y')

print(instance('123', a = '456'))
print(instance(final = False))
