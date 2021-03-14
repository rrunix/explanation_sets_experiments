import numpy as np


def extract_anchor_explanations(folds_explanations):
    explanations = []
    for fold_id, fold_data in enumerate(folds_explanations):
        for explanation in fold_data:
            explanations.append({
                'sample_idx': explanation['sample_idx'],
                'time': explanation['time'],
                'fold_id': fold_id,
                'anchor': explanation['explanation'].data['raw']['names'],
                'precision': float(explanation['explanation'].data['precision']),
                'coverage': float(explanation['explanation'].data['coverage']),
                'num_conditions': len(explanation['explanation'].data['raw']['names']),
                'own_coverage': float(explanation['explanation'].data['raw']['own_coverage']),
                'own_precision': float(explanation['explanation'].data['raw']['own_precision']),
                'prediction': round(float(explanation['explanation'].data['raw']['prediction'])),
            })
    
    return explanations


def extract_counterfactual_explanations(folds_explanations):
    explanations = []
    for fold_id, fold_data in enumerate(folds_explanations):
        for explanation in fold_data:
            explanations.append({
                'sample_idx': explanation['sample_idx'],
                'time': explanation['time'],
                'fold_id': fold_id,
                'actual': explanation['explanation']['actual'],
                'counterfactual_class': explanation['explanation']['counterfactual_class'],
                'counterfacutal': explanation['explanation']['counterfactual'],
                'observation': explanation['explanation']['observation'],
                'loss': explanation['explanation']['loss'],
                'belongs_neighborhood': explanation['explanation']['belongs_neighborhood'],
                'meets_grouping_measure': explanation['explanation']['meets_grouping_measure']
            })
    
    return explanations


def add_counterfactual_predictions(base_folder, target_folder):
    folds_data = joblib.load(os.path.join(base_folder, 'folds.bz2'))
    target_folder = os.path.join(base_folder, target_folder)
    for fold in os.listdir(target_folder):
        fold_id = int(fold.replace('fold_', ''))
        fold_folder = os.path.join(target_folder, fold)
        
        target_column = folds_data[fold_id]['dataset_info'].target_column
        model = folds_data[fold_id]['model']
        
        for sample in os.listdir(fold_folder):
            file_folder = os.path.join(fold_folder, sample)
            exp = joblib.load(file_folder)
            exp['actual'] = folds_data[fold_id]['df'].loc[exp['sample_idx'], tc]
            exp['counterfactual_class'] =  model.predict(exp['explanation']['counterfactual'])[0]
            
            joblib.dump(exp, file_folder)

            
def mean_std(x):
    return f"{round(np.mean(x), 2)} ({round(np.std(x), 2)})"


def set_execution_mode(use_spawn):
    if use_spawn:
        import multiprocessing
        multiprocessing.set_start_method('spawn')
