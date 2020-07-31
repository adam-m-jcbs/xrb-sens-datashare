"""
Define exceptions for remote interaction
"""

class RemoteException(Exception):
    def __str__(self):
        return ' [REMOTE] ERROR: exception {}.'.format(*self.args)
class RemoteCommandNotFoundError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: commnand "{}" not found'.format(*self.args)
    __repr__ = __str__
class RemoteExecutionError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: executing {}(*{}, **{}). {}'.format(*self.args)
class RemoteActionError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: unknown action "{}".'.format(*self.args)
class RemoteMissingAttributeError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: missing attribute'.format(*self.args)
class RemoteMissingValueError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: missing value'.format(*self.args)
class RemoteAttributeError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: unknown attribute "{}".'.format(*self.args)
class RemoteMethodError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: trying to return remote method "{}".'.format(*self.args)
class RemoteIndexError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: invalid index {} for attribute "{}".'.format(*self.args)
class RemoteRefIndexError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: invalid ref_index {} of {} for attribute "{}".'.format(*self.args)
class RemoteComplexObjectError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: complex object "{}" cannot be returned.'.format(*self.args)
class RemoteNumpyArrayError(RemoteComplexObjectError):
    def __str__(self):
        return ' [REMOTE] INFO: numpy object "{}" will not be returned.'.format(*self.args)
class ConnectionError(RemoteException):
    def __str__(self):
        return ' [REMOTE] ERROR: Server not connected.'.format(*self.args)
