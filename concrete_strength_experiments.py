import glob
import sys
from experiments_utils import set_execution_mode
import joblib
from itertools import chain
import pandas as pd
from mlexpies import neighborhood
from mlexpies.explainers import counterfactual, anchor, explainer
from mlexpies import grouping_measures
import experiments_runner
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable tensorflow logging


USE_SPAWN = False

# Running and global args
DEFAULT_BASE_ARGS = {'ids_slice': None, 'jobs': 1}

CC_EXPLAINER_ARGS = {'not_meet_loss': 30, 'max_evals': 50}
ANCHOR_ARGS = {'beam_size': 1}

BASE_FOLDER = 'experiments/concrete_strength/'
RANDOM_STATE = 0
REWRITE = True


def counterfactual_explanations(name, folds_data, distance, grouping_measure, base_args, rewrite=False, radius=float('inf')):
    export_file = os.path.join(BASE_FOLDER, name + '.bz2')
    partial_explanations_cache_file = os.path.join(
        BASE_FOLDER, name + '_cache')
    if not os.path.exists(export_file) or rewrite:
        explanations = experiments_runner.execute(
            counterfactual.CounterfactualExplainer(grouping_measure,
                                                   neighborhood.NeighborhoodFactory(
                                                       distance,
                                                       radius
                                                   ),
                                                   **CC_EXPLAINER_ARGS
                                                   ), folds_data, partial_explanations_cache_file, **base_args)
        joblib.dump(explanations, export_file)


def anchor_explanations(name, folds_data, distance, grouping_measure, base_args, rewrite=False, radius=float('inf')):
    export_file = os.path.join(BASE_FOLDER, name + '.bz2')
    partial_explanations_cache_file = os.path.join(
        BASE_FOLDER, name + '_cache')

    if not os.path.exists(export_file) or rewrite:
        explanations = experiments_runner.execute(
            anchor.AnchorExplainer(grouping_measure,
                                   neighborhood.NeighborhoodFactory(
                                       distance,
                                       radius
                                   ),
                                   anchor_params=ANCHOR_ARGS,
                                   random_state=RANDOM_STATE), folds_data, partial_explanations_cache_file, **base_args)
        joblib.dump(explanations, export_file)


def run_experiments(base_args):

    # Load model data
    folds_data = joblib.load(os.path.join(BASE_FOLDER, 'folds.bz2'))

    # Radius
    penalization_radius = 30

    # Base distance
    base_distance = neighborhood.GowerNeighborhoodDistance()

    # Counterfactual base grouping measure
    radius_small_grouping_measure = grouping_measures.RadiusGroupingMeasure(
        radius=5)

    # Manifold closeness
    manifold_closeness = neighborhood.ManifoldCloseness(distance_not_meet=penalization_radius)
    gower_with_manifold = neighborhood.AdditiveNeighborhoodDistancesChain([base_distance, manifold_closeness])

    # Base explanations. Groups only equals predictions

    # Counterfactual-based explanations
    # counterfactual_explanations('cc_regression_rsmall@sgower', folds_data, base_distance,
    #                             grouping_measures.Negate(radius_small_grouping_measure), base_args=base_args, rewrite=REWRITE)
    #
    # # Semifactual-based explanations
    # anchor_explanations('sf_regression_rsmall@sgower', folds_data,
    #                     base_distance, radius_small_grouping_measure, base_args=base_args, rewrite=REWRITE)

    # Now, two predictions are grouping if they distance (L1-norm) is less than or equals to 1
    radius_big_grouping_measure = grouping_measures.RadiusGroupingMeasure(
        radius=10)

    # # Counterfactual-based explanations
    # counterfactual_explanations('cc_regression_rbig@sgower', folds_data, base_distance,
    #                             grouping_measures.Negate(radius_big_grouping_measure), base_args=base_args, rewrite=REWRITE)
    #
    # # Semifactual-based explanations
    # anchor_explanations('sf_regression_rbig@sgower', folds_data,
    #                     base_distance, radius_big_grouping_measure, rewrite=REWRITE)

    # Finally, since increasing the quality is the desired outcome, we only consider counterfactuals whose prediction is greater than that of the
    # observation of interest.
    greather_than_grouping = grouping_measures.GreatherThanGroupingMeasure()
    # counterfactual_explanations('cc_regression_gt@sgower', folds_data, base_distance,
    #                             greather_than_grouping, base_args=base_args, rewrite=REWRITE)

    greather_than_grouping_offset = grouping_measures.GreatherThanGroupingMeasure(
        offset=5)
    # counterfactual_explanations('cc_regression_gt_offset@sgower', folds_data, base_distance,
    #                             greather_than_grouping_offset, base_args=base_args, rewrite=REWRITE)


    counterfactual_explanations('cc_regression_gt_manifold@sgower', folds_data, gower_with_manifold,
                                greather_than_grouping, base_args=base_args, rewrite=REWRITE, radius=penalization_radius)

    counterfactual_explanations('cc_regression_gt_offset_manifold@sgower', folds_data, gower_with_manifold,
                               greather_than_grouping_offset, base_args=base_args, rewrite=REWRITE, radius=penalization_radius)

    counterfactual_explanations('cc_regression_rbig_manifold@sgower', folds_data, gower_with_manifold,
                                grouping_measures.Negate(radius_big_grouping_measure), base_args=base_args,
                                rewrite=REWRITE, radius=penalization_radius)

    counterfactual_explanations('cc_regression_rsmall_manifold@sgower', folds_data, gower_with_manifold,
                                 grouping_measures.Negate(radius_small_grouping_measure), base_args=base_args,
                                rewrite=REWRITE, radius=penalization_radius)


if __name__ == '__main__':
    set_execution_mode(USE_SPAWN)
    base_args = DEFAULT_BASE_ARGS.copy()

    if len(sys.argv) == 3:
        base_args['ids_slice'] = slice(
            int(sys.argv[1]), int(sys.argv[2]), None)

    run_experiments(base_args)
