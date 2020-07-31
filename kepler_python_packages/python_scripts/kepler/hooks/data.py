"""
Some demonstration examples
"""

# for class to be transferrable to another process, possibly another
# machine, any files should only be opened on first __call__
class DataSaver():
    def __init__(self, filename, variables, debug = False):
        self.variables = variables
        self.filename = filename
        self._debug = debug

    def __call__(self, kepler):
        with open(self.filename, 'at') as f:
            res = []
            for v in self.variables:
                r = getattr(kepler, v)
                if self._debug:
                    print(f'[mess] {v}: {r}')
                res.append(r)
            if self._debug:
                print(f'[mess] {res}')
            f.write(f'{res}\n')
