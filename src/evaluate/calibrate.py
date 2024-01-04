# Types
from pathlib import Path

# Calibration
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd
import numpy as np
import pickle

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os

#context can be: paper, notebook, talk, poster
sns.set_theme(context="poster", palette="pastel", style="ticks", font_scale=0.8)

def train_and_store_calibration_model(cfg, image_results_path, results_name) -> None:
    """
    Get the calibration model for the given results file and store it.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration file.
    image_results_path : Path
        Path to the image results folder.
    results_name : str
        Name of the results file.

    Returns
    -------
    None
    """
    # Progress
    if cfg.verbose:
        print("\n===============================================")
        print(f"Training Calibration Model on {results_name}")
        print("===============================================")

    # Load results
    results_df = pd.read_csv(image_results_path / results_name)

    # Extract information about true positives vs false postives
    all_true = []
    all_predicted = []
    id_cols = sorted([col for col in results_df.columns if col.startswith("id_")])
    for id_col in id_cols[:-1]:
        id = int(id_col.split("_")[1])
        y_true = results_df[f"id_{id}"] == results_df[f"id_{id+1}"]
        y_prob = results_df[f"p{id}_{id+1}"].to_numpy()
        all_predicted = all_predicted + y_prob.tolist()
        all_true = all_true + y_true.tolist()

    # Fit calibration model
    model = IsotonicRegression(y_min=cfg.evaluate.y_min, 
                               y_max=cfg.evaluate.y_max, 
                               out_of_bounds=cfg.evaluate.out_of_bounds, 
                               increasing=cfg.evaluate.increasing)
    model.fit(all_predicted, all_true)

    # Store calibration model
    if cfg.evaluate.calibration_model_name is not None:
        model_name = cfg.evaluate.calibration_model_name
    else:
        model_name = cfg.experiment_name

    model_path = cfg.calibration_model_dir
    if cfg.verbose:
        print(f"Storing calibration model as {Path(model_path) / Path(model_name)}")
    pickle.dump(model, open(Path(model_path) / Path(model_name + ".pkl"), "wb"))


def save_calibration_plot(cfg, results_dir, title="Calibration Plot", frame_shift=0, result_type="Unfiltered"):
    """
    Create a calibration plot for the given results dataframe.
    """
    # Progress
    if cfg.verbose:
        print("\n=========================================")
        print("Creating Calibration Plot - ", result_type)
        print("=========================================\n")

    if result_type == "unfiltered":
        file_name_start = "unfiltered_trajectories"
        fig_name = f"calibration_plot_unfiltered.png"
    elif result_type == "filtered":
        file_name_start = "filtered_trajectories"
        fig_name = f"calibration_plot_filtered.png"
    elif result_type == "dropped":
        file_name_start = "dropped_trajectories"
        fig_name = f"calibration_plot_dropped_trajectories.png"
    elif result_type == "dropped_merging":
        file_name_start = "dropped_merging_trajectories"
        fig_name = f"calibration_plot_dropped.png"
    else:
        raise NotImplementedError("Result type not implemented. In calibration_plot.py")

    # Load simulated data
    for file in os.listdir(results_dir):
        if file.startswith(file_name_start):
            results_df_name = file                
            results_df = pd.read_csv(results_dir / results_df_name)

            # Check if results are empty
            if results_df.shape[0] == 0:
                print(f"Results for {result_type} are empty. Skipping calibration plot.")
                return

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

    ax.set_title(f"{title}: {result_type}", fontweight="bold")
    fig.savefig(results_dir / fig_name, dpi=100)
    

