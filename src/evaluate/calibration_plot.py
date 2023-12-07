import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import seaborn as sns
from seaborn.categorical import boxplot

#context can be: paper, notebook, talk, poster
sns.set_theme(context="poster", palette="pastel", style="ticks", font_scale=0.8)


def save_calibration_plot(cfg, results_dir):
    """
    Create a calibration plot for the given results dataframe.
    """
    # Progress
    if cfg.verbose:
        print("\n=========================================")
        print("Creating Calibration Plot")
        print("=========================================\n")

    # Get results df
    name = "results_" + cfg.experiment_name + ".csv"
    results_df = pd.read_csv(results_dir / name)

    # Prepare plot for calibration curve
    fig, ax = plt.subplots(figsize=(9, 7))

    # Extract information about true positives vs false postives and plot calibration curve
    all_predicted = []
    id_cols = sorted([col for col in results_df.columns if col.startswith("id_")])
    for id_col in id_cols[:-1]:
        id = int(id_col.split("_")[1])
        y_true = results_df[f"id_{id}"] == results_df[f"id_{id+1}"]
        y_prob = results_df[f"p{id}_{id+1}"].to_numpy()
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=8, strategy="quantile"
        )
        all_predicted = all_predicted + y_prob.tolist()
        ax.plot(mean_predicted_value, fraction_of_positives, label=f"{id}-{id+1}")

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
    ax.set_ylabel("Accuracy", fontsize=17)
    ax.set_xlabel("Decision Threshold", fontsize=17)

    # Make ticks smaller
    ax.tick_params(axis='both', which='both', labelsize=14)
 
    # Put probability distribution label for last plot on top
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-2:] + handles[:-2]
    labels = labels[-2:] + labels[:-2]

    ax.legend(handles, labels, fontsize=13, loc="upper left")


    # Set title
    if cfg.generate_results.uncertainty_type == "scaled_entries":
        type = "Scaled OT Matrix Entries"

    elif cfg.generate_results.uncertainty_type == "scaled_ranks":
        type = "Scaled OT Matrix Ranks"

    else:
        raise NotImplementedError("Uncertainty type not implemented. In get_trajectories.py")
    
    ax.set_title(f"Calibration Plot: {type}", fontweight="bold")


    fig.savefig(results_dir / "calibration_plot.png", dpi=100)

    return None
