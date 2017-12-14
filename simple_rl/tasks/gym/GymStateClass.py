# Python imports
import numpy

# Local imports
from simple_rl.mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class GymState(State):
    ''' Gym State class '''

    def __init__(self, data=[], is_terminal=False,interaction_features=False):
        self.data = data
        State.__init__(self, data=data, is_terminal=is_terminal)
        self.interaction_features=interaction_features
        
#     def features(self):
#         if self.interaction_features:
#             a=self.data
#             b=self.data
#             c=numpy.outer(a,b)
#             inds=numpy.triu_indices(len(a))
#             d=c[inds]
#             return numpy.array(c).flatten()
#         else:
#             return numpy.array(self.data).flatten()
            
    
