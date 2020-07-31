"""
Collection of specific excptions
"""

class KeplerTerminated(Exception):
    """
    Exception raised when KEPLER terminates.
    """
    def __init__(self, code = None):
        self.code = code

    def __str__(self):
        """Return error message."""
        if self.code is None:
            return ("\n [KEPLER] terminated.\n")
        else:
            return ("\n [KEPLER] terminated with code {:d}\n").format(
                self.code)


class KeplerNotRunning(Exception):
    def __str__(self):
        return ' [Kepler] Not running.'
