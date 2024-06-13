import pandas as pd
import numpy as np
import wandb
import os
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


class TrackingEvaluation:
    """
    Simple class for evaluating optimal transport results.
    ----------
    Parameters:
    cfg: DictConfig:
        Global config.
    results_path: str:
        Path to directory where results are stored.
    result_type: str:
        Type of results to evaluate. Must be either 'Unfiltered' or 'Filtered' or 'Dropped' or 'Dropped_Merging'.

    Attributes:
    args: object:
        Evaluation arguments.
    results_path: str:
        Path to directory where results are stored.
    verbose: bool:
        Flag indicating whether to print progress messages.
    wandb: bool:
        Flag indicating whether to log results to wandb.
    result_type: str:
        Type of results to evaluate. Must be either 'Unfiltered' or 'Filtered' or 'Dropped' or 'Dropped_Merging'.
    results_df: pandas.DataFrame:
        DataFrame containing the results.
    file_name_suffix: str:
        Suffix of the results file.

    Methods:
    compute_and_store_scores():
        Compute and store scores.
    """

    # Class variable for storing ot solution
    # ot = None # for OLD CODE

    def __init__(self, cfg, results_path, result_type="Unfiltered"):
        self.args = cfg.evaluate
        self.results_path = results_path
        self.verbose = cfg.verbose
        self.wandb = cfg.wandb
        self.result_type = result_type

        if result_type == "unfiltered":
            file_name_start = "unfiltered_trajectories"
        elif result_type == "filtered":
            file_name_start = "filtered_trajectories"
        elif result_type == "dropped":
            file_name_start = "dropped_trajectories"
        elif result_type == "dropped_merging":
            file_name_start = "dropped_merging"
        elif result_type == "renamed":
            file_name_start = "filtered_trajectories_y0_x0_basic_renamed"
        else:
            raise NotImplementedError(
                "Result type not implemented. Must be either 'Unfiltered' or 'Filtered' or 'Dropped' or 'Dropped_Merging'"
            )

        # Load simulated data
        for file in os.listdir(self.results_path):
            if file.startswith(file_name_start):
                results_df_name = file
                # Get suffix of results_df_name
                self.file_name_suffix = results_df_name.split("_")[-1].split(".")[0]

                self.results_df = pd.read_csv(self.results_path / results_df_name)

                self.num_trajectories = len(self.results_df)
        # OLD CODE
        # self.image_ot_path = image_ot_path
        # self.simulated_data = SimulatedData(cfg, image_simulated, None)
        # self.position_df = pd.read_csv(image_simulated, index_col=0)

    def compute_and_store_scores(self):
        # Progress
        if self.verbose:
            print("\n=========================================")
            print(f"Computing Scores for OT - {self.result_type}")
            print("=========================================\n")
            print(f"Currently processing:")

        # Dictionary for storing scores
        scores = []

        # Extract information about true-positives vs false-positives and plot calibration curve
        results_df = self.results_df

        # Check if results are empty
        if results_df.shape[0] == 0:
            print(
                f"Results for {self.result_type} are empty. Skipping computation of scores."
            )
            return

        # List of k values to compute precision
        k_values = self.args.precision_at_k

        # For total auroc, auprc
        y_true_all = []
        y_prob_all = []

        # For metrics on whole trajectories
        y_complete = [True] * self.num_trajectories

        # step for wandb plots
        step = 0

        # Retrieve frame ids from results df
        id_cols = sorted(
            [
                int(col.split("_")[1])
                for col in results_df.columns
                if col.startswith("id_")
            ]
        )
        for id_col in id_cols[:-1]:
            # Dictionary for storing scores for current frame
            scores_frame = {}

            # verbose
            if self.verbose:
                print(f"\nFrames: {id_col} - {id_col+1}")

            # store ids
            scores_frame[f"frames"] = f"{id_col}-{id_col+1}"

            # get true positives
            y_true = results_df[f"id_{id_col}"] == results_df[f"id_{id_col+1}"]
            y_prob = results_df[f"p{id_col}_{id_col+1}"].to_numpy()
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)

            y_complete = np.logical_and(y_complete, y_true)

            # get auprc, and auroc
            if self.args.auprc:
                auprc = average_precision_score(y_true, y_prob)
                scores_frame[f"auprc"] = round(auprc, 3)

                if self.verbose:
                    print(f"auprc: {auprc}")

            if self.args.auroc:
                auroc = roc_auc_score(y_true, y_prob)

                scores_frame[f"auroc"] = round(auroc, 3)

                if self.verbose:
                    print(f"auroc: {auroc}")

            if self.args.brier:
                brier_score = brier_score_loss(y_true, y_prob)

                scores_frame[f"brier"] = round(brier_score, 3)

                if self.verbose:
                    print(f"brier: {brier_score}")

            if self.args.precision_at_k:
                # get precision@k
                k_values = self.args.k_values
                k_values = [k for k in k_values if k <= self.num_trajectories]

                precision_at_k = []
                sorted_probs_indices = y_prob.argsort()

                for k in k_values:
                    # check for -1 case
                    if k == -1:
                        k = len(y_true)

                    # get top k indices
                    top_k_indices = sorted_probs_indices[-k:]

                    # deal with ties
                    k_eff = len(top_k_indices)

                    # get precision@k
                    precision_at_k.append((y_true[top_k_indices].sum() / k_eff))

                    # store precision@k
                    scores_frame[f"p@_{k}"] = round(precision_at_k[-1], 3)

                    if self.verbose:
                        print(f"p@_{k}: {precision_at_k[-1]}")

            # store scores for current frame
            if self.wandb:
                for key, value in scores_frame.items():
                    if key != "frames":
                        wandb.log({key: value}, step=step)
                step += 1

            scores.append(scores_frame)

        # Save scores
        scores_df = pd.DataFrame.from_records(scores)

        # Compute total auroc, auprc
        if self.args.auprc or self.args.auroc:
            y_true_all = np.concatenate(y_true_all)
            y_prob_all = np.concatenate(y_prob_all)

        if self.args.auprc:
            auprc = average_precision_score(y_true_all, y_prob_all)
            scores_df[f"auprc_total"] = round(auprc, 3)

            if self.verbose:
                print(f"\nAll Frames:\nauprc_total: {auprc}")

            if self.wandb:
                step += 1
                wandb.log({"auprc_total": auprc}, step=step)

        if self.args.auroc:
            auroc = roc_auc_score(y_true_all, y_prob_all)
            scores_df[f"auroc_total"] = round(auroc, 3)

            if self.verbose:
                print(f"auroc_total: {auroc}")

            if self.wandb:
                step += 1
                wandb.log({"auroc_total": auroc}, step=step)

        if self.args.brier:
            brier_score = brier_score_loss(y_true_all, y_prob_all)
            scores_df[f"brier_total"] = round(brier_score, 3)

            if self.verbose:
                print(f"brier_total: {brier_score}")

            if self.wandb:
                wandb.log({"brier_total": brier_score}, step=step)

        # Compute average of each column
        avg_row = scores_df.mean(numeric_only=True)
        avg_row = ["Mean"] + [round(i, 3) for i in avg_row]

        max_row = scores_df.max(numeric_only=True)
        max_row = ["Max"] + [round(i, 3) for i in max_row]

        scores_df.loc["avg"] = avg_row
        scores_df.loc["max"] = max_row

        if self.result_type == "unfiltered":
            scores_df.to_csv(self.results_path / "unfiltered_scores__.csv", index=False)

        elif self.result_type == "filtered":
            scores_df.to_csv(
                self.results_path
                / Path("filtered_scores__" + self.file_name_suffix + ".csv"),
                index=False,
            )

        elif self.result_type == "dropped":
            scores_df.to_csv(
                self.results_path
                / Path("dropped_scores__" + self.file_name_suffix + ".csv"),
                index=False,
            )

        elif self.result_type == "dropped_merging":
            scores_df.to_csv(
                self.results_path
                / Path("dropped_merging_scores__" + self.file_name_suffix + ".csv"),
                index=False,
            )

        elif self.result_type == "renamed":
            scores_df.to_csv(
                self.results_path
                / Path("renamed_scores__" + self.file_name_suffix + ".csv"),
                index=False,
            )

            # separate metrics for complete trajectories
            traj_accuracy = y_complete.sum() / len(y_complete)
            traj_accuracy = round(traj_accuracy, 3)

            # save metrics for complete trajectories
            with open(
                self.results_path
                / Path("renamed_scores__" + self.file_name_suffix + ".txt"),
                "w",
            ) as f:
                f.write(f"Trajectory Accuracy: {traj_accuracy}")

        else:
            raise ValueError(
                "result_type must be either 'Unfiltered' or 'Filtered' or 'Dropped' or 'Dropped_Merging'"
            )
