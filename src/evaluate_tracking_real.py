import pandas as pd

from evaluate.get_scores import TrackingEvaluation
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RESULTS_PATH = "data/05_results"
gt_results = ["ch01-20-full", "ch02-10-full", "ch02-14-full", "ch02-16-full"]

times = ["05", "15", "30"]

algo_results = [
    [
        "20240110IL_DropletTracking_PBMCBeadline_Chamber01-20_MedMov_LargeDrop_"
        + t
        + "min"
        for t in times
    ],
    [
        "20240110IL_DropletTracking_PBMCBeadline_Chamber02-10_SmallMov_Sparse_"
        + t
        + "min"
        for t in times
    ],
    [
        "20240110IL_DropletTracking_PBMCBeadline_Chamber02-14_MedMov_SparseLargeDrops_"
        + t
        + "min"
        for t in times
    ],
    [
        "20240110IL_DropletTracking_PBMCBeadline_Chamber02-16_LargeMov_Shift_"
        + t
        + "min"
        for t in times
    ],
]
# overwrite for new experiments
RESULTS_PATH = "data/05_results_Luca"
gt_results = [
    "Exp3_01_long_no",
    "Exp3_02_long_no_medium",
    "Exp3_03_long_small_medium_large",
]
algo_results = [
    ["Exp3_01_short_no"],
    ["Exp3_02_short_no_medium"],
    ["Exp3_03_short_small_large"],
]


def compute__frame_metrics(matching):

    # Precision at k
    k_values = [100, 500, 1000, 2000, 5000, -1]
    precision_at_k = {k: 0 for k in k_values}
    for k in k_values:
        precision_at_k[k] = len(matching[matching["rank"] <= k]) / k

    # AUPRC
    auprc = average_precision_score(matching["gt"], matching["score"])

    # AUROC
    auroc = roc_auc_score(matching["gt"], matching["score"])

    # Brier score
    brier = brier_score_loss(matching["gt"], matching["score"])

    return precision_at_k, auprc, auroc, brier


if __name__ == "__main__":
    """
    Main function for evaluating tracking.
    This is mainly a workaround to still be able to use the TrackingEvaluation class.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    None
    """

    result_type = "renamed"

    for gt_res, algo_res_list in tqdm(zip(gt_results, algo_results)):
        for algo_res in tqdm(algo_res_list, desc=f"GT: {gt_res}"):

            ### Old code
            # Load simulated data
            results_path = f"{RESULTS_PATH}/{algo_res}/"
            results_path = Path(results_path)
            cfg = {
                "evaluate": {
                    "precision_at_k": True,
                    "k_values": [100, 500, 1000, 2000, 5000, -1],
                    "auprc": True,
                    "auroc": True,
                    "brier": True,
                },
                "verbose": True,
                "wandb": False,
            }

            # turn into DictConfig
            cfg = DictConfig(cfg)

            ot_evaluation = TrackingEvaluation(
                cfg, results_path, result_type=result_type
            )
            ot_evaluation.compute_and_store_scores()
