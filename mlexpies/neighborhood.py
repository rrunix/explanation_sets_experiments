import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


class NeighborhoodFactory:

    def __init__(self, distance, radius=float('inf')):
        self.distance  = distance
        self.radius = radius

    def create(self, sample):
        return Neighborhood(sample, distance=self.distance, radius=self.radius)


class Neighborhood:

    def __init__(self, sample, distance, radius=float('inf')):
        self.sample = sample
        self.radius = radius
        self.base_distance = distance

    def distance(self, other):
        if len(other.shape) == 2:
            return np.array([self.base_distance.distance(self.sample, x) for x in self._iterrows(other)])
        else:
            return self.base_distance.distance(self.sample, other)

    def belongs_neighborhood(self, other):
        return self.distance(other) < self.radius

    def _iterrows(self, other):
        if isinstance(other, pd.DataFrame):
            return (v for _, v in other.iterrows())
        else:
            return other

    def fit(self, X_train, dataset_info, neighborhood_fit_params=None):
        self.base_distance.fit(X_train, dataset_info, neighborhood_fit_params=neighborhood_fit_params)

    def optimize_search_space(self, search_space):
        """
        Optimizes (in-place) the search space provided.

        The optimized search space might contain elements that are not contained in the neighborhood;
        however, all values in the neigborhood are contained in the optimized search space.

        Args:
            search_space: Search space to optimize

        Returns:
            optimized search space
        """
        return self.base_distance.optimize_search_space(self.sample, search_space, self.radius)


class NeighborhoodDistance:

    def distance(self, center, other):
        raise NotImplemented

    def fit(self, X_train, dataset_info, neighborhood_fit_params=None):
        pass
    
    def optimize_search_space(self, center, search_space, radius):
        return search_space


class ZeroNeighborhoodDistance(NeighborhoodDistance):
    def distance(self, center, other):
        return 0


class RestrictedNeighborhoodDistance(NeighborhoodDistance):

    def __init__(self, feature, cmp='eq', distance_not_meet=float('inf')):
        self.feature = feature
        self.cmp = cmp
        self.distance_not_meet = distance_not_meet

        if self.cmp not in ('eq', 'leq', 'geq'):
            raise ValueError(f"Invalid comparison mode{self.cmp}")


    def distance(self, center, other):
        sample_value = center[self.feature]
        other_value = other[self.feature]

        if self.cmp == 'eq':
            meet = sample_value == other_value
        elif self.cmp == 'leq':
            meet = sample_value <= other_value
        elif self.cmp == 'geq':
            meet = sample_value >= other_value

        return 0 if meet else self.distance_not_meet 

    def optimize_search_space(self, center, search_space, radius):
        if self.cmp == 'eq':
            search_space.set_feature_value(
                self.feature, center[self.feature])
        elif self.cmp == 'leq':
            search_space.update_numeric_bound(
                self.feature, high=center[self.feature])
        elif self.cmp == 'geq':
            search_space.update_numeric_bound(
                self.feature, low=center[self.feature])
        return search_space


class AdditiveNeighborhoodDistancesChain(NeighborhoodDistance):

    def __init__(self, distances):
        self.distances = distances
    
    def distance(self, center, other):
        distance_sum = self.distances[0].distance(center, other)

        for distance in self.distances[1:]:
            distance_sum += distance.distance(center, other)

        return distance_sum

    def optimize_search_space(self, center, search_space, radius):
        for distance in self.distances:
            search_space = distance.optimize_search_space(center, search_space, radius)

        return search_space

    def fit(self, X_train, dataset_info, neighborhood_fit_params=None):
        for distance in self.distances:
            distance.fit(X_train, dataset_info, neighborhood_fit_params=neighborhood_fit_params)


class GowerNeighborhoodDistance(NeighborhoodDistance):

    def __init__(self, sparse=True):
        self.sparse = sparse

    def fit(self, X_train, dataset_info, neighborhood_fit_params=None):
        self.dataset_info_ = dataset_info
        self.feature_ranges = {feature: {'low': X_train[feature].min(), 'high': X_train[feature].max()} 
                               for feature in self.dataset_info_.numerical_features}

    def distance(self, center, other):
        check_is_fitted(self)

        distance = 0
        changes = 0

        for feature in self.dataset_info_.categorical_features:
            sample_value = center[feature]
            other_value = other[feature]
            changed = int(sample_value != other_value)
            distance += changed
            changes += changed

        for feature in self.dataset_info_.numerical_features:
            info = self.feature_ranges[feature]

            sample_value = center[feature]
            other_value = other[feature]
            diff = (np.abs(sample_value - other_value)) / info['high']
            distance += diff
            changes += 1 if abs(diff) > 0.001 else 0

        if self.sparse:
            return distance / (1 + self.dataset_info_.num_features - changes)
        else:
            return distance / self.dataset_info_.num_features


class DiversityDistance(NeighborhoodDistance):

    def __init__(self, base_distance):
        self.base_distance = base_distance

    def fit(self, X_train, dataset_info, neighborhood_fit_params=None):
        self.penalize_obs_ = neighborhood_fit_params['penalize_obs']
        self.base_distance.fit(X_train, dataset_info, neighborhood_fit_params)

    def distance(self, center, other):
        check_is_fitted(self)
        
        dist = 1 / (1+self.base_distance.distance(center, self.penalize_obs_)) + 1 / (1+self.base_distance.distance(self.penalize_obs_, other))
        dist = dist + self.base_distance.distance(center, other)
        return dist
