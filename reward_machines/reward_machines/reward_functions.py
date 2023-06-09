#Adapted from 
#https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/reward_machines/reward_functions.py

import numpy as np

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")

class LinearRewardFunction(RewardFunction):
    """
    Defines a linear reward that is a linear function of 
    feature values defined in s_info. 
    """
    def __init__(self,v,c):
        super().__init__()
        self.v = v #vector of coefficients
        self.c = c #scalar constant 

    def get_type(self):
        return "linear"
    
    def get_reward(self, s_info):
        return np.dot(self.v, list(s_info.values())) + self.c 

class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c
