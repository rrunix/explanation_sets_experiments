import numpy as np
import pandas as pd
from copy import deepcopy


class DatasetInfo:

    def __init__(self, target_column, feature_order, real_features=(), integer_features=(), categorical_map=None):
        self.feature_order = feature_order
        self.target_column = target_column
        self.real_features = real_features
        self.integer_features = integer_features
        self.categorical_map = categorical_map or {}
        self.numerical_categorical_map = {self.features.index(k): [k+"@{"+vv+"}" for vv in v] for k, v in self.categorical_map.items()}

    @property
    def categorical_features(self):
        return list(self.categorical_map.keys())

    @property
    def numerical_features(self):
        return list(self.real_features) + list(self.integer_features)

    @property
    def features(self):
        return self.feature_order

    @property
    def num_features(self):
        return len(self.feature_order)

    def __str__(self):
        return str(self.features_definition)
