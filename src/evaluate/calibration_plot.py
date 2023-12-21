import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from pathlib import Path
import seaborn as sns
from seaborn.categorical import boxplot
import os

#context can be: paper, notebook, talk, poster
sns.set_theme(context="poster", palette="pastel", style="ticks", font_scale=0.8)


def save_calibration_plot(cfg, results_dir, title="Calibration Plot", frame_shift=0, result_type="Unfiltered"):
    """
    Create a calibration plot for the given results dataframe.
    """
    # Progress
    if cfg.verbose:
        print("\n=========================================")
        print("Creating Calibration Plot - ", result_type)
        print("=========================================\n")

    if result_type == "Unfiltered":
        file_name_start = "results"
        fig_name = "calibration_plot.png"

    elif result_type == "Filtered":
        file_name_start = "filtered_results"

    else:
        raise NotImplementedError("Result type not implemented. In calibration_plot.py")

    # Load simulated data
    for file in os.listdir(results_dir):
        if file.startswith(file_name_start):
            results_df_name = file
            # Get suffix of results_df_name
            file_name_suffix = results_df_name.split("_")[-1].split(".")[0]

            if result_type == "Filtered":
                fig_name = f"calibration_plot_{file_name_suffix}.png"
                
            results_df = pd.read_csv(results_dir / results_df_name)

    # Prepare fig
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract information about true positives vs false postives and plot calibration curve
    all_true = []
    all_predicted = []
    id_cols = sorted([col for col in results_df.columns if col.startswith("id_")])
    for id_col in id_cols[:-1]:
        id = int(id_col.split("_")[1])
        y_true = results_df[f"id_{id}"] == results_df[f"id_{id+1}"]
        y_prob = results_df[f"p{id}_{id+1}"].to_numpy()

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=8, strategy="uniform"
        )

        print(f'{id}-{id+1}', brier_score_loss(y_true, y_prob))
        all_predicted = all_predicted + y_prob.tolist()
        all_true = all_true + y_true.tolist()
        ax.plot(mean_predicted_value, fraction_of_positives, label=f"{id+frame_shift}-{id+frame_shift+1}")

    # Only 3 decimal places
    bs = brier_score_loss(all_true, all_predicted)
    bs = round(bs, 3)
    textstr = f'Total BS = {bs}'
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='grey', alpha=0.2)

    # place a text box in upper left in axes coords
    ax.text(0.03, 0.68, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # Get total brier score
    bs = brier_score_loss(all_true, all_predicted)
    print("Total brier score:", bs)

    # Get counts in bins
    counts, bins = np.histogram(all_predicted, bins=15)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    # Normalize
    counts = (15/10)*counts / len(all_predicted)

    # plot
    ax.plot(bins[1:], counts, label="Probability Distribution", linestyle="--", color="grey")
    ax.fill_between(bins[1:], counts, 0, color='grey', alpha=.1)

    # Add legend and save plot
    ax.set_ylabel("Observation", fontsize=13)
    ax.set_xlabel("Prediction", fontsize=13)

    # Make ticks smaller
    ax.tick_params(axis='both', which='both', labelsize=14)

    # Put probability distribution label for last plot on top
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-2:] + handles[:-2]
    labels = labels[-2:] + labels[:-2]

    ax.legend(handles, labels, fontsize=12, loc="upper left")

    # Set title
    # if cfg.generate_results.uncertainty_type == "scaled_entries":
    #     type = "Scaled OT Matrix Entries"

    # elif cfg.generate_results.uncertainty_type == "scaled_ranks":
    #     type = "Scaled OT Matrix Ranks"
    # else:
    #     raise NotImplementedError("Uncertainty type not implemented. In get_trajectories.py")
    
    ax.set_title(f"Calibration Plot: {result_type}", fontweight="bold")
    fig.savefig(results_dir / fig_name, dpi=100)