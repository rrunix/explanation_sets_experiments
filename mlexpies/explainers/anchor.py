import numpy as np
import pandas as pd
from alibi.explainers.anchor_tabular import AnchorTabular
from sklearn.utils.validation import check_is_fitted

from mlexpies.explainers import explainer

def build_pandas_query(anchor_names, dataset_info):
    query = " and ".join(anchor_names)
    query = query.replace(' = ', ' == ')
    
    for k_categs in dataset_info.numerical_categorical_map.values():
        for i, categ in enumerate(k_categs):
            query = query.replace(categ, str(i))
            
    return query


class AnchorExplainer(explainer.Explainer):

    def __init__(self, grouping_measure, neighborhood_factory=None, anchor_params=None, min_train=2500, random_state=0):
        self.grouping_measure_impl = grouping_measure
        self.anchor_params = anchor_params or {}
        self.random_state = random_state
        self.neighborhood_factory = neighborhood_factory
        self.min_train = min_train

    def fit(self, model, X_train, dataset_info):
        self.model_ = model
        self.X_train_ = X_train
        self.dataset_info_ = dataset_info
        
    def grouping_measure(self, factual_pred, other):
        return self.grouping_measure_impl(factual_pred, other, factual_pred=factual_pred)

    def _calculate_initial_sample(self, neighborhood):
        belong_neighborhood_mask =  neighborhood.belongs_neighborhood(self.X_train_)
        X_train_neighborhood = self.X_train_[belong_neighborhood_mask]

        if len(X_train_neighborhood) < self.min_train:
            weights = neighborhood.distance(self.X_train_)

            if abs(weights.min() - weights.max()) < 0.1:
                # Uniform distribution
                weights = np.ones_like(weights) / len(weights)
            else:
                weights = abs(weights - weights.max() + weights.min())
                weights = weights / weights.sum()

            sample = self.X_train_.sample(n=self.min_train - len(X_train_neighborhood), weights=weights)
            X_train_neighborhood = pd.concat((X_train_neighborhood, sample))

        return X_train_neighborhood

    def explain(self, sample):
        check_is_fitted(self)
        y_sample = self.model_.predict(sample.values.reshape(1, -1)).reshape(-1)[0]

        neighborhood = self.neighborhood_factory.create(sample)
        neighborhood.fit(self.X_train_, self.dataset_info_)

        def predict_fn(x):
            grouped = self.grouping_measure(y_sample, self.model_.predict(x))

            if isinstance(grouped, pd.Series):
                grouped = grouped.values


            x_df = pd.DataFrame(x, columns=sample.index)
            in_neighboorhood = neighborhood.belongs_neighborhood(x_df)

            if isinstance(in_neighboorhood, pd.Series):
                in_neighboorhood = in_neighboorhood.values

            grouped &= in_neighboorhood

            return grouped

        
        X_train = self._calculate_initial_sample(neighborhood)

        explainer = AnchorTabular(predict_fn, X_train.columns, seed=self.random_state, categorical_names=self.dataset_info_.numerical_categorical_map)
        explainer.fit(X_train.values)
        sample_exp = explainer.explain(sample.values, **self.anchor_params)

        # Filter instances that do not belong to the neighborhood
        belong_neighborhood_mask =  neighborhood.belongs_neighborhood(self.X_train_)
        X_train_neighborhood = self.X_train_[belong_neighborhood_mask]

        return self._generate_explanation(X_train_neighborhood, sample_exp, predict_fn, y_sample)

    def _generate_explanation(self, X_train, sample_exp, predict_fn, y_sample):
        query_anchor = build_pandas_query(sample_exp.data['raw']['names'], self.dataset_info_)
        covered_samples = X_train if len(query_anchor) == 0 else X_train.query(query_anchor)
        sample_exp.data['raw']['own_coverage'] = (len(covered_samples) / len(X_train)) if len(X_train) > 0 else 0
        sample_exp.data['raw']['prediction'] = y_sample

        if len(covered_samples) == 0:
            sample_exp.data['raw']['own_precision'] = np.NaN
        else:
            sample_exp.data['raw']['own_precision'] = np.sum(predict_fn(covered_samples)) / len(covered_samples)
   
        return sample_exp

def extract_anchor_explanations(folds_explanations):
    explanations = []
    for fold_data in folds_explanations.values():
        for explanation in fold_data['explanations']:
            explanations.append({
                'sample_idx': explanation['sample_idx'],
                'time': explanation['time'],
                'anchor': explanation['explanation'].data['raw']['names'],
                'precision': float(explanation['explanation'].data['precision']),
                'coverage': float(explanation['explanation'].data['coverage']),
                'num_conditions': len(explanation['explanation'].data['raw']['names']),
                'own_coverage': float(explanation['explanation'].data['raw']['own_coverage']),
                'own_precision': float(explanation['explanation'].data['raw']['own_precision']),
                'prediction': float(explanation['explanation'].data['raw']['prediction']),
            })

    return explanations


def anchor_contains_feature(anchor, feature_list):
    for condition in anchor:
        for feature in feature_list:
            if feature in condition:
                return True
    return False
