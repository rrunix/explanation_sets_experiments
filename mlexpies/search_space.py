from copy import deepcopy


class SearchSpace:

    def __init__(self, df, dataset_info):
        self.dataset_info = dataset_info
        self.search_space = {} 

        for feature in self.dataset_info.features:
            if feature in self.dataset_info.categorical_features:
                self.search_space[feature] = {
                    'name': feature, 'type': 'categorical', 'choices': list(df[feature].unique())}
            else:
                self.search_space[feature] = {'name': feature, 'type': 'real', 'low': df[feature].min(), 'high': df[feature].max()}

    def update_numeric_bound(self, feature, high=None, low=None):
        if not self.is_numeric_feature(feature):
            raise ValueError(f"Feature {feature} is not numerical")

        feat_space = self.search_space[feature]

        if high is not None:
            feat_space['high'] = min(feat_space['high'], high)

        if low is not None:
            feat_space['low'] = max(feat_space['low'], low)

    def set_feature_value(self, feature, value):
        feat_space = self.search_space[feature]

        if self.is_numeric_feature(feature):
            if value < feat_space['low'] or value > feat_space['high']:
                raise ValueError(
                    f"Cannot set value {value} because it is outside the current bounds ({feat_space['low']}, {feat_space['high']}).")

            feat_space['low'] = value
            feat_space['high'] = value

        else:
            if value not in feat_space['choices']:
                raise ValueError(
                    f"Cannot set value {value} because it is not available in the current choices {feat_space['choices']}")

            feat_space['choices'] = [value]

    def remove_choice(self, feature, choice):
        if self.is_numeric_feature(feature):
            raise ValueError(
                f"Cannot remove choice because feature {feature} is numerical")

        if choice in feat_space['choices']:
            feat_space['choices'].remove(choice)

    def is_numeric_feature(self, feature):
        return self.search_space[feature]['type'] in ('real', 'integer')

    def __str__(self):
        return str(self.search_space)
