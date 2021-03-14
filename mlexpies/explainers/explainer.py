from sklearn.utils.validation import check_is_fitted

class Explainer:

    def fit(self, model, X_train, dataset_info):
        pass

    def explain(self, sample):
        pass