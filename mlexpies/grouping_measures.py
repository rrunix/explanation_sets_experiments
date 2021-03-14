import numpy as np


class BinaryGroupingMeasure:

    def __init__(self, group_equal):
        self.group_equal = group_equal

    def __call__(self, x, y, factual_pred=None):
        if self.group_equal:
            return (np.round(x) == np.round(y)).astype(int)
        else:
            return (np.round(x) != np.round(y)).astype(int)


class RadiusGroupingMeasure:

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, x, y, factual_pred=None):
        return (abs(np.round(x) - np.round(y)) <= self.radius).astype(int)


class GreatherThanGroupingMeasure:
    """ This grouping measure is not symmetric
    """
    def __init__(self, offset=0):
        self.offset = offset

    def __call__(self, x, y, factual_pred=None):
        if factual_pred:
            return ((np.round(x + self.offset) > np.round(factual_pred)) & (np.round(y + self.offset) > np.round(factual_pred))).astype(int)
        return (np.round(x + self.offset) < np.round(y)).astype(int)
    
    
class Negate:
    
    def __init__(self, measure):
        self.measure = measure

    def __call__(self, x, y, factual_pred=None):
        return np.mod(self.measure(x, y, factual_pred = factual_pred) + 1, 2)
