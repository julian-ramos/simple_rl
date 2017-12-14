# Python imports
import numpy

# Local imports
from simple_rl.mdp.StateClass import State
import pandas as pd

''' DataStateClass.py: Contains a State class for Data. '''

class DataState(State):
    ''' Data State class '''
        
    def __init__(self, data,is_terminal=False):
        State.__init__(self, data=data, is_terminal=is_terminal)
        
    def features(self):
        a=self.data
        b=self.data
        c=numpy.outer(a,b)
        inds=numpy.triu_indices(len(a))
        d=c[inds]
        return np.array(c).flatten()
