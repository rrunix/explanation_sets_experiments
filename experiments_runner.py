import multiprocessing as mp
from contexttimer import Timer
import tempfile
import joblib
import logging
from multiprocessing import Pool
from itertools import chain
from tqdm import tqdm
import joblib
import os
import warnings
import random
import threading
from pathlib import Path
import time


class DisableLogger():
    #https://stackoverflow.com/a/20251235
    def __enter__(self):
        logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


class ProgressBar(threading.Thread):

    def __init__(self, cache_dir, total=None, sleep_time=2):
        super().__init__()
        self.sleep_time = sleep_time
        self._running = True
        self.cache_dir = cache_dir
        self.total = total

    def terminate(self):
        self._running = False

    def run(self):
        tq = tqdm(total=self.total)
        current = 0

        while self._running and (self.total is None or current < self.total):
            if os.path.exists(self.cache_dir):
                new_current = len(list(Path(self.cache_dir).rglob('**/*.bz2')))

                if current < new_current:
                    tq.update(new_current - current)
                    tq.refresh()

                current = new_current
                time.sleep(self.sleep_time)

        tq.close()


def fold_execute(args):
    fold_id, fold_data, explainer, explain_ids, random_state, cache_dir = args
    df = fold_data['df']
    train_idx = fold_data['train_idx']
    test_idx = fold_data['test_idx']
    model = fold_data['model']
    dataset_info = fold_data['dataset_info']


    cache_explanations_dir = os.path.join(cache_dir, f"fold_{fold_id}")
    os.makedirs(cache_explanations_dir, exist_ok=True)

    df_train = df.drop(columns=dataset_info.target_column)
    X_train = df_train.loc[train_idx]

    explanations = []
    explainer.fit(model, X_train, dataset_info)

    for sample_idx in explain_ids:
        explanation_cache_dir = os.path.join(cache_explanations_dir, f"sample_{sample_idx}.bz2")

        try:
            if os.path.exists(explanation_cache_dir):
                explanation = joblib.load(explanation_cache_dir)
            else:
                sample = df_train.loc[sample_idx]

                with DisableLogger(), Timer() as t, warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sample_exp = explainer.explain(sample)

                explanation = {
                    'sample_idx': sample_idx,
                    'explanation': sample_exp,
                    'time': t.elapsed
                }
                joblib.dump(explanation, explanation_cache_dir)

            explanations.append(explanation)
        except Exception as e:
            print(e)

    return explanations


def execute(explainer, folds_data, cache_dir, jobs=None, ids_slice=None, random_state=0):
    args = []
    total_exps = 0
    print(f"Starting {cache_dir} experiment....")

    for fold_id, fold_data in enumerate(folds_data):
        total_exps += len(fold_data['test_idx'])

        if ids_slice is not None:
            explain_ids = list(fold_data['test_idx'])[ids_slice]
        else:
            explain_ids = fold_data['test_idx']

        args.append((fold_id, fold_data, explainer, explain_ids, random_state, cache_dir))

    pbar = ProgressBar(cache_dir, total=total_exps)
    pbar.start()

    if jobs == 1:
        res = [fold_execute(iter_args) for iter_args in args]
    else:
        with Pool(processes=jobs) as p:
            res = p.map(fold_execute, args)

    pbar.terminate()
    print(f"{cache_dir} done!")
    return res
