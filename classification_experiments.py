from experiments_utils import set_execution_mode
import experiments_runner
from mlexpies import grouping_measures
from mlexpies.explainers import counterfactual, anchor
from mlexpies import neighborhood
import pandas as pd
from itertools import chain
import joblib
import glob
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable tensorflow logging


USE_SPAWN = False

# Running and global args
DEFAULT_BASE_ARGS = {'ids_slice': None, 'jobs': 1}

# Not meet radius
NOT_MEET_RADIUS = 30

CC_EXPLAINER_ARGS = {'not_meet_loss': 20, 'max_evals': 100}
ANCHOR_ARGS = {'beam_size': 1}

BASE_FOLDER = 'experiments/adult/'
RANDOM_STATE = 0
REWRITE = True


class AnchorCounterfactual(anchor.AnchorExplainer):

    def __init__(self, partial_experiments_folder, *args, **kwargs):
        self.partial_experiments_folder = partial_experiments_folder
        super().__init__(*args, **kwargs)

    def explain(self, sample):
        sample_idx = sample.name
        explanation_cache_dir = list(glob.glob(
            f"{self.partial_experiments_folder}/*/sample_{sample_idx}.bz2", recursive=True))
        if len(explanation_cache_dir) > 0:
            assert len(
                explanation_cache_dir) == 1, f"Duplicated index {explanation_cache_dir} {len(explanation_cache_dir)}"

            cc_data = joblib.load(explanation_cache_dir[0])
            cc_sample = cc_data['explanation']['counterfactual']
            cc_sample = cc_sample.loc[sample.index]
            return super().explain(cc_sample)
        else:
            raise ValueError(
                f'Counterfactual explanation for sample {sample_idx} not found in {self.partial_experiments_folder}')


class DiversityCounterfactual(counterfactual.CounterfactualExplainer):

    def __init__(self, partial_experiments_folder, *args, **kwargs):
        self.partial_experiments_folder = partial_experiments_folder
        super().__init__(*args, **kwargs)

    def explain(self, sample):
        sample_idx = sample.name
        explanation_cache_dir = list(glob.glob(
            f"{self.partial_experiments_folder}/*/sample_{sample_idx}.bz2", recursive=True))
        if len(explanation_cache_dir) > 0:
            assert len(
                explanation_cache_dir) == 1, f"Duplicated index {explanation_cache_dir} {len(explanation_cache_dir)}"

            cc_data = joblib.load(explanation_cache_dir[0])
            cc_sample = cc_data['explanation']['counterfactual']
            cc_sample = cc_sample.loc[sample.index]
            neighborhood_fit_params = {'penalize_obs': cc_sample}
            return super().explain(sample, neighborhood_fit_params=neighborhood_fit_params)
        else:
            raise ValueError(
                f'Counterfactual explanation for sample {sample_idx} not found in {self.partial_experiments_folder}')


def classification_counterfactual_explanations(name, folds_data, distance, rewrite=False, radius=NOT_MEET_RADIUS, base_args=DEFAULT_BASE_ARGS):
    export_file = os.path.join(BASE_FOLDER, name + '.bz2')
    partial_explanations_cache_file = os.path.join(
        BASE_FOLDER, name + '_cache')
    if not os.path.exists(export_file) or rewrite:
        explanations = experiments_runner.execute(
            counterfactual.CounterfactualExplainer(grouping_measures.BinaryGroupingMeasure(group_equal=False),
                                                   neighborhood.NeighborhoodFactory(
                distance,
                radius
            ),
                **CC_EXPLAINER_ARGS
            ), folds_data, partial_explanations_cache_file, **base_args)
        joblib.dump(explanations, export_file)


def diversity_classification_counterfactual_explanations(name, counterfactuals_folder, folds_data, distance, rewrite=False, radius=NOT_MEET_RADIUS, base_args=DEFAULT_BASE_ARGS):
    export_file = os.path.join(BASE_FOLDER, name + '.bz2')
    partial_explanations_cache_file = os.path.join(
        BASE_FOLDER, name + '_cache')

    partial_cc_folder = os.path.join(
        BASE_FOLDER, counterfactuals_folder + '_cache')

    if not os.path.exists(export_file) or rewrite:
        explanations = experiments_runner.execute(
            DiversityCounterfactual(partial_cc_folder, grouping_measures.BinaryGroupingMeasure(group_equal=False),
                                    neighborhood.NeighborhoodFactory(
                distance,
                radius
            ),
                **CC_EXPLAINER_ARGS
            ), folds_data, partial_explanations_cache_file, **base_args)
        joblib.dump(explanations, export_file)


def classification_anchor_explanations(name, folds_data, distance, rewrite=False, radius=NOT_MEET_RADIUS, base_args=DEFAULT_BASE_ARGS):
    export_file = os.path.join(BASE_FOLDER, name + '.bz2')
    partial_explanations_cache_file = os.path.join(
        BASE_FOLDER, name + '_cache')

    if not os.path.exists(export_file) or rewrite:
        explanations = experiments_runner.execute(
            anchor.AnchorExplainer(grouping_measures.BinaryGroupingMeasure(group_equal=True),
                                   neighborhood.NeighborhoodFactory(
                                       distance,
                                       radius
            ),
                anchor_params=ANCHOR_ARGS,
                random_state=RANDOM_STATE), folds_data, partial_explanations_cache_file, **base_args)
        joblib.dump(explanations, export_file)


def cs_anchor_explanations(name, counterfactuals_folder, folds_data, distance, rewrite=False, radius=NOT_MEET_RADIUS, base_args=DEFAULT_BASE_ARGS):
    export_file = os.path.join(BASE_FOLDER, name + '.bz2')
    partial_explanations_cache_file = os.path.join(
        BASE_FOLDER, name + '_cache')

    partial_cc_folder = os.path.join(
        BASE_FOLDER, counterfactuals_folder + '_cache')

    if not os.path.exists(export_file) or rewrite:
        explanations = experiments_runner.execute(
            AnchorCounterfactual(partial_cc_folder,
                                 grouping_measures.BinaryGroupingMeasure(
                                     group_equal=True),
                                 neighborhood.NeighborhoodFactory(
                                     distance,
                                     radius
                                 ),
                                 anchor_params=ANCHOR_ARGS,
                                 random_state=RANDOM_STATE), folds_data, partial_explanations_cache_file, **base_args)
        joblib.dump(explanations, export_file)


def run_experiments(base_args=DEFAULT_BASE_ARGS):

    # # Load model data
    folds_data = joblib.load(os.path.join(BASE_FOLDER, 'folds.bz2'))

    # Base distance
    base_distance = neighborhood.GowerNeighborhoodDistance()

    # Base explanations. Neighborhood includes all instances from the feature space.
    # Semifactual-based explanations
    classification_anchor_explanations(
        'sf_clasification_base', folds_data, base_distance, rewrite=REWRITE, base_args=base_args)

    # Counterfactual-based explanations
    classification_counterfactual_explanations(
        'cc_clasification_base@new', folds_data, base_distance, rewrite=REWRITE, base_args=base_args)

    # Counterfactual sets
    cs_anchor_explanations('cs_clasification_base', 'cc_clasification_base',
                           folds_data, base_distance, rewrite=REWRITE, base_args=base_args)

    # Diversity counterfactuals
    diversity_distance = neighborhood.DiversityDistance(base_distance)
    diversity_classification_counterfactual_explanations(
        'cc_div_clasification_base', 'cc_clasification_base', folds_data, diversity_distance, rewrite=REWRITE, base_args=base_args)

    # Restric explanations that involve changes over Age or MaritalStatus
    restrict_distance_single = neighborhood.AdditiveNeighborhoodDistancesChain([
        neighborhood.RestrictedNeighborhoodDistance(
            feature, cmp='eq', distance_not_meet=NOT_MEET_RADIUS)
        for feature in ['Age', 'Race', 'Sex', 'MaritalStatus', 'Relationship']
    ] + [base_distance])

    restrict_distance_set = neighborhood.AdditiveNeighborhoodDistancesChain([
        neighborhood.RestrictedNeighborhoodDistance(
            feature, cmp='eq', distance_not_meet=NOT_MEET_RADIUS)
        for feature in ['Race', 'Sex', 'MaritalStatus', 'Relationship']
    ] + [base_distance])

    # Counterfactual-based explanations
    classification_counterfactual_explanations(
        'cc_clasification_restrict@sgower', folds_data, restrict_distance_single, rewrite=REWRITE, base_args=base_args)

    # Counterfactual set explanations
    cs_anchor_explanations('cs_clasification_restrict@sgower', 'cc_clasification_restrict@sgower',
                           folds_data, restrict_distance_set, rewrite=REWRITE, base_args=base_args)

    # Semifactual-based explanations
    classification_anchor_explanations(
        'sf_clasification_restrict@sgower', folds_data, restrict_distance_set, rewrite=REWRITE)


if __name__ == '__main__':
    set_execution_mode(USE_SPAWN)

    base_args = DEFAULT_BASE_ARGS.copy()

    if len(sys.argv) == 3:
        base_args['ids_slice'] = slice(
            int(sys.argv[1]), int(sys.argv[2]), None)

    run_experiments(base_args)
