from mlexpies.dataset import DatasetInfo
from mlexpies.search_space import SearchSpace
from mlexpies.explainers import explainer

from hyperopt import fmin, hp, tpe, atpe, Trials
import pandas as pd
import numpy as np
import random


class CounterfactualLoss:

    def __init__(self, model, sample, grouping_measure, neighborhood, dataset_info, not_meet_loss, fixed_choices={}):
        self.model = model
        self.neighborhood = neighborhood
        self.grouping_measure = grouping_measure
        self.not_meet_loss = not_meet_loss
        self.dataset_info = dataset_info
        self.y = model.predict(sample.astype(float)).reshape(-1)[0]
        self.fixed_choices = fixed_choices
        self.n_calls = 0

    def to_obs(self, params):
        x = np.array([params[k] if k in params else self.fixed_choices[k] for k in self.dataset_info.features])
        return x
 
    def meets_grouping_measure(self, params):
        y_hat = self.model.predict(self.to_obs(params).reshape(1, -1)).reshape(-1)[0]
        return self.grouping_measure(self.y, y_hat) == 1

    def batch_meets_grouping_measure(self, df):
        df = df[self.dataset_info.features]
        y_hat = self.model.predict(df)
        return self.grouping_measure(self.y, y_hat) == 1

    def belongs_neighborhood(self, params):
        return self.neighborhood.belongs_neighborhood(pd.Series(params))

    def __call__(self, params):
        self.n_calls += 1
        if not self.belongs_neighborhood(params) or not self.meets_grouping_measure(params):
            return self.not_meet_loss
        
        return self.neighborhood.distance(pd.Series(params))


class CounterfactualExplainer(explainer.Explainer):

    def __init__(self, grouping_measure, neighborhood_factory, not_meet_loss=10, max_evals=300, evaluate_dataset=True, max_warmup_size=2500):
        self.grouping_measure = grouping_measure
        self.neighborhood_factory = neighborhood_factory
        self.not_meet_loss = not_meet_loss
        self.max_evals = max_evals
        self.evaluate_dataset = evaluate_dataset
        self.fitted = False
        self.max_warmup_size = max_warmup_size

    def _convert_search_space(self, search_space):
        res = {}
        fixed_values = {}
        for col, info in search_space.search_space.items():
            if not search_space.is_numeric_feature(col):
                res[col] = hp.choice(col, info['choices'])
                if len(info['choices']) == 1:
                    fixed_values[col] = info['choices']
            else:
                if info['high'] == info['low']:
                    res[col] = info['low']
                    fixed_values[col] = info['low']
                else:
                    if info['type'] == 'integer':
                        res[col] = hp.quniform(col, info['low'], info['high'])
                    else:
                        res[col] = hp.uniform(col, info['low'], info['high'])

        return res, fixed_values

    def fit(self, model, X_train, dataset_info):
        self.model_ = model
        self.X_train_ = X_train
        self.X_train_dict_ = X_train.to_dict(orient='records')
        self.dataset_info_ = dataset_info

    def _convert_back(self, search_space, best):
        best = best.copy()
        for feature in self.dataset_info_.features:
            info = search_space.search_space[feature]

            if feature in self.dataset_info_.categorical_features:
                best_idx = best[feature]
                best[feature] = info['choices'][best_idx]
            elif info['low'] == info['high']:
                best[feature] = info['low']
        return best

    def _convert_to_hp(self, search_space, obs):
        if len(self.dataset_info_.categorical_features) > 0:
            hp_dict = obs.copy()
            for feature in self.dataset_info_.categorical_features:
                feature_value = hp_dict[feature]
                choices = search_space.search_space[feature]['choices']
                if feature_value in search_space.search_space[feature]['choices']:
                    hp_dict[feature] = choices.index(hp_dict[feature])
                else:
                    hp_dict[feature] = random.choice(list(range(len(choices))))

            for feature in self.dataset_info_.numerical_features:
                feature_value = hp_dict[feature]
                feat_info = search_space.search_space[feature]
                low, high = feat_info['low'], feat_info['high']

                if feature_value < low:
                    hp_dict[feature] = low
                elif feature_value > high:
                    hp_dict[feature] = high
            return hp_dict
        else:
            return obs

    def _calculate_initial_sample(self, search_space, loss):
        if self.evaluate_dataset:
            meet_grouping_measure = loss.batch_meets_grouping_measure(self.X_train_)
            weights = loss.neighborhood.distance(self.X_train_)
            weights[~meet_grouping_measure] += self.not_meet_loss
            weights = abs(weights - weights.max() + weights.min())
            weights = weights / weights.sum()
            np.random.seed(0)
            initial_sample = np.random.choice(self.X_train_dict_, self.max_warmup_size, p=weights)
            return [self._convert_to_hp(search_space, obs) for obs in initial_sample]

        return None

    def explain(self, sample, neighborhood_fit_params=None):
        explainer.check_is_fitted(self)
        search_space = SearchSpace(self.X_train_, self.dataset_info_)

        neighborhood = self.neighborhood_factory.create(sample)
        neighborhood.fit(self.X_train_, self.dataset_info_, neighborhood_fit_params)

        search_space = neighborhood.optimize_search_space(search_space)
        converted_search_space, fixed_choices = self._convert_search_space(search_space)
        loss = CounterfactualLoss(self.model_, sample, self.grouping_measure,
                                  neighborhood, self.dataset_info_, not_meet_loss=self.not_meet_loss, fixed_choices=fixed_choices)
        
        best = None
        extraction_method = ''
        if self.max_evals > 0:
            points_to_evaluate = self._calculate_initial_sample(search_space, loss)

            best = fmin(fn=loss, space=converted_search_space, algo=tpe.suggest, max_evals=self.max_evals, 
                show_progressbar=False, points_to_evaluate=points_to_evaluate)
            best = self._convert_back(search_space, best)
            extraction_method = 'bayesian search'

        if (self.max_evals == 0 or best is None) or not loss.meets_grouping_measure(best):
            meet_grouping_measure = loss.batch_meets_grouping_measure(self.X_train_)
            if meet_grouping_measure.sum() > 0:
                valid = self.X_train_[meet_grouping_measure]
                best_idx = loss.neighborhood.distance(valid).argmin()
                best = valid.iloc[best_idx]
                best = best.to_dict()
                extraction_method = 'closest enemy'

        if best is None:
            best = {}
            belongs_neighborhood = False
            meets_grouping_measure = False
            loss_v = np.inf
        else:          
            belongs_neighborhood = loss.belongs_neighborhood(best)
            meets_grouping_measure = loss.meets_grouping_measure(best)
            loss_v = loss(best)
        
        explanation = {
            'actual': loss.y,
            'counterfactual_class': self.model_.predict(loss.to_obs(best).reshape(1, -1))[0],
            'observation': sample,
            'counterfactual': pd.Series(best),
            'loss': loss_v,
            'belongs_neighborhood': belongs_neighborhood,
            'meets_grouping_measure': meets_grouping_measure,
            'extraction_method': extraction_method
        }
        #print(explanation)
        return explanation
