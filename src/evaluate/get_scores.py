import pandas as pd 
import numpy as np
import wandb
import os

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

class OtEvaluation():
    """
    Simple class for running some OT analyses
    """
    # Class variable for storing ot solution
    # ot = None # for OLD CODE

    def __init__(self, cfg, image_simulated, image_ot_path, results_path, result_type="Unfiltered"):
        self.args = cfg.evaluate
        self.results_path = results_path
        self.verbose = cfg.verbose
        self.wandb = cfg.wandb
        self.result_type = result_type

        if result_type == "Unfiltered":
            file_name_start = "results"
        elif result_type == "Filtered":
            file_name_start = "filtered_results"
        else:
            raise NotImplementedError("Result type not implemented. In calibration_plot.py")

        # Load simulated data
        for file in os.listdir(self.results_path):
            if file.startswith(file_name_start):
                results_df_name = file
                self.results_df = pd.read_csv(self.results_path / results_df_name)

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
            print(f'Currently processing:')

        # Dictionary for storing scores
        scores = []

        # Extract information about true positives vs false postives and plot calibration curve
        results_df = self.results_df

        # List of k values to compute accuracy
        k_values = self.args.accuracy_k

        # For total auroc, auprc
        y_true_all = []
        y_prob_all = []

        # step for wandb plots
        step = 0

        # Retrieve frame ids from results df
        id_cols = sorted([col for col in results_df.columns if col.startswith("id_")])
        for id_col in id_cols[:-1]:
            # Dictionary for storing scores for current frame
            scores_frame = {}

            # get current transition ids
            id = int(id_col.split("_")[1])

            # verbose
            if self.verbose:
                print(f"\nFrames: {id} - {id+1}")

            # store ids
            scores_frame[f"frames"] = f"{id}-{id+1}"

            # get true positives
            y_true = results_df[f"id_{id}"] == results_df[f"id_{id+1}"]
            y_prob = results_df[f"p{id}_{id+1}"].to_numpy()
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
    
            # get auprc, and auroc
            if self.args.auprc:
                auprc = average_precision_score(y_true, y_prob)
                scores_frame[f"auprc"] = auprc

                if self.verbose:
                    print(f'auprc: {auprc}')

            if self.args.auroc:
                auroc = roc_auc_score(y_true, y_prob)
                scores_frame[f"auroc"] = auroc

                if self.verbose:
                    print(f'auroc: {auroc}')

            if self.args.brier:
                brier_score = brier_score_loss(y_true, y_prob)
                scores_frame[f"brier"] = brier_score

                if self.verbose:
                    print(f'brier: {brier_score}')

            if self.args.accuracy:
                # TODO: check whether sklearn top_k_accuracy is nicer/better
                # get top k accuracy
                k_values = self.args.accuracy_k
                top_k_accuracy = []
                sorted_probs_indices = y_prob.argsort()
                
                for k in k_values:
                    # check for -1 case
                    if k == -1:
                        k = len(y_true)

                    # get top k indices
                    top_k_indices = sorted_probs_indices[-k:]

                    # deal with ties
                    k_eff = len(top_k_indices)

                    # get top k accuracy
                    top_k_accuracy.append((y_true[top_k_indices].sum() / k_eff))

                    # store top k accuracy
                    scores_frame[f"{k}_accuracy"] = top_k_accuracy[-1]

                    if self.verbose:
                        print(f'{k}_accuracy: {top_k_accuracy[-1]}')

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
            scores_df[f"auprc_total"] = auprc

            if self.verbose:
                print(f'\nAll Frames:\nauprc_total: {auprc}')

            if self.wandb:
                step += 1
                wandb.log({"auprc_total": auprc}, step=step)
        
        if self.args.auroc:
            auroc = roc_auc_score(y_true_all, y_prob_all)
            scores_df[f"auroc_total"] = auroc

            if self.verbose:
                print(f'auroc_total: {auroc}')

            if self.wandb: 
                step += 1
                wandb.log({"auroc_total": auroc}, step=step)

        if self.args.brier:
            brier_score = brier_score_loss(y_true_all, y_prob_all)
            scores_df[f"brier_total"] = brier_score

            if self.verbose:
                print(f'brier_total: {brier_score}')
            
            if self.wandb:
                wandb.log({"brier_total": brier_score}, step=step)


        # Compute average of each column
        avg_row = scores_df.mean(numeric_only=True)
        avg_row = ["Mean"] + [i for i in avg_row] 

        max_row = scores_df.max(numeric_only=True)
        max_row = ["Max"] + [i for i in max_row] 

        scores_df.loc['avg'] = avg_row
        scores_df.loc['max'] = max_row

        if self.result_type == "Unfiltered":
            scores_df.to_csv(self.results_path / "scores.csv", index=False)

        elif self.result_type == "Filtered":
            scores_df.to_csv(self.results_path / "scores_filtered.csv", index=False)

        else:
            raise ValueError("result_type must be either 'Unfiltered' or 'Filtered'")

