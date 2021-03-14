import os
import joblib

from alibi.datasets import fetch_adult
import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score



from mlexpies.dataset import DatasetInfo


RANDOM_STATE = 0
CV = 10


def create_lgb_model(experiment_folder, df, dataset_info, model_params, num_folds, score_fn, 
                     random_state = RANDOM_STATE):
    kfold = KFold(n_splits=CV, shuffle=True, random_state=random_state)
    folds = kfold.split(df.index)
    target_column = dataset_info.target_column
    folds_data = []

    os.makedirs(experiment_folder, exist_ok=True)
    
    for i, (train_idx, test_idx) in enumerate(folds):
        df_train = df.drop(columns=target_column)
        X_train = df_train.loc[train_idx]
        y_train = df.loc[train_idx, target_column]

        model = lgbm.train(model_params, 
                           lgbm.Dataset(X_train, y_train), 
                           categorical_feature=list(dataset_info.categorical_features))

        score = score_fn(df.loc[test_idx, target_column], model.predict(df_train.loc[test_idx]))
        folds_data.append({
            'df': df,
            'train_idx': list(train_idx),
            'test_idx': list(test_idx),
            'model': model,
            'dataset_info': dataset_info,
            'score': score
        })
        
        print(f"Fold {i}: score {score}")

    joblib.dump(folds_data, os.path.join(experiment_folder, 'folds.bz2'))
    
    
    
def concrete_strength_experiment():
    df = pd.read_csv('datasets/concrete_data.csv')
    df.rename(columns={
        'Cement (component 1)(kg in a m^3 mixture)': 'cement',
        'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'blast_furnace_slag',
        'Fly Ash (component 3)(kg in a m^3 mixture)': 'fly_ash',
        'Water  (component 4)(kg in a m^3 mixture)': 'water',
        'Superplasticizer (component 5)(kg in a m^3 mixture)': 'superplasticizer',
        'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'coarse_aggregate',
        'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'fine_aggregate', 
        'Age (day)': 'age',
        'Concrete compressive strength(MPa, megapascals) ': 'compressive_strength'
    }, inplace=True)
    
    target_column = 'compressive_strength'

    regression_parameters = {
        'random_state': RANDOM_STATE,
        'nthreads': 1,
        'metric': 'mse',
        'objective': 'regression',
        'force_row_wise': True,
        'verbose': 0
    }

    features = [col for col in df.columns if col != target_column]
    regression_dataset_info = DatasetInfo(
        target_column,
        features,
        real_features=features
    )
    
    create_lgb_model('experiments/concrete_strength/', df, regression_dataset_info, regression_parameters, CV, 
                 score_fn=mean_squared_error)
    
    
def adult_experiment():
    adult = fetch_adult()
    data = adult.data
    target = adult.target
    feature_names = [feature.title().replace(' ', '') for feature in adult.feature_names]
    category_map = {feature_names[k]: v for k, v in adult.category_map.items()}
    integer_features = [feature for feature in feature_names if feature not in category_map]

    df = pd.DataFrame(data=data, columns=feature_names)

    target_column = 'IsLess50k'
    df[target_column] = target
    df = df[df['Country'] == 9].copy()
    df.reset_index(drop=True, inplace=True)
    
    classification_parameters = {
    'random_state': RANDOM_STATE,
    'nthreads': 1,
    'metric': 'binary_logloss',
    'objective': 'binary',
    'force_row_wise': True,
    'verbose': 0
    }


    classification_dataset_info = DatasetInfo(
        'IsLess50k',
        [col for col in df.columns if col != 'IsLess50k'],
        real_features=(),
        integer_features=integer_features,
        categorical_map=category_map
    )
    
    def classification_score_fn(y, y_hat):
        return f1_score(y, np.round(y_hat))
    
    create_lgb_model('experiments/adult/', df, classification_dataset_info, classification_parameters, CV, 
                 score_fn=classification_score_fn)
    
    
    
if __name__ == '__main__':
    concrete_strength_experiment()
    adult_experiment()