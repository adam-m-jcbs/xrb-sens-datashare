import numpy as np
from ..datainterface import DataInterface
from .base import BasePlot

class DataPlot(BasePlot):
    def __init__(self,
                 data = None,
                 **kwargs):

        assert isinstance(data, DataInterface), 'need to provide data source interface DataInterface'
        self.data = data

        np.seterr(under = 'ignore')

        super().__init__(**kwargs)
