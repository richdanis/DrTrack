
# Types
from pathlib import Path
from omegaconf import DictConfig

# Data handling
import pandas as pd
import numpy as np
import math

# System
import sys
import os
from datetime import datetime

# Progress bar
from tqdm.auto import tqdm

# Plotting
import matplotlib.pyplot as plt
from matplotlib.path import Path as matplotlibPath
from matplotlib.widgets import LassoSelector
from matplotlib.lines import Line2D

sys.path.append('src/')
from preprocess.raw_image_reader import get_image_cut_as_ndarray

PROJECT_PATH = Path(os.getcwd())
DATA_PATH = Path(PROJECT_PATH / "data")
RESULT_PATH = Path(DATA_PATH / "05_results")

SAVE_KEY = "c"
SWAPPING_KEY = "a"
CUTTING_KEY = "w"
CONFIRMATION_KEY = "enter"

# Auxiliary stuff for being able to swap edges
swap_mode = False
selected_for_swapping = []
selected_for_swapping_frames = []
auxiliary_swap_lines_drawn = []
auxiliary_swap_lines = []

# Auxiliary stuff for being able to cut edges
cut_mode = False
selected_for_cutting_trajectory = None
selected_for_cutting_frame = None
auxiliary_cut_lines_drawn = None

# Global variables needed in order to update the dataset from inside the callback functions
results_df_x = None
results_df_y = None
results_df = None
all_lines = None

# Global variables to keep track of where the trajectories begin (not trivial for when we have partial trajectories)
trajectory_beginning = None

# Set aspect ratio to 1
plt.gca().set_aspect('equal')

def launch_interactive_explorer(cfg: DictConfig, 
                        image_path: Path, 
                        results_path: Path, 
                        image_name: str, 
                        y_stride: int, 
                        x_stride: int) -> None:
    """
    Start the interactive explorer.

    Parameters
    ----------
    cfg : DictConfig
        The configuration file.
    image_path : str
        The path to the image.
    results_path : str
        The path to the results.
    image_name : str
        The name of the image.
    y_stride : int
        The y stride.
    x_stride : int
        The x stride.

    Returns
    -------
    None
    """
    # Must declare to use the global variables
    global results_df_x
    global results_df_y
    global results_df
    global all_lines
    global trajectory_beginning

    # Int32 is a special pandas datatype, which is a integer 32 bit datatype plus it can have value NAN
    results_df = pd.read_csv(results_path)

    # Assert if dataframe empty
    assert not results_df.empty, "The dataframe is empty."

    # There is no real need to drop the NA rows
    results_df['discard'] = True

    if cfg.frame_range is not None:
        # Extract the relevant frames
        first_frame = cfg.frame_range[0]
        last_frame = cfg.frame_range[1]
        x_pos_cols = [i for i in results_df.columns if str(i).startswith("x") and int(i[1:]) >= first_frame and int(i[1:]) <= last_frame]
        y_pos_cols = [i for i in results_df.columns if str(i).startswith("y") and int(i[1:]) >= first_frame and int(i[1:]) <= last_frame]
    else:
        x_pos_cols = [i for i in results_df.columns if str(i).startswith("x")]
        y_pos_cols = [i for i in results_df.columns if str(i).startswith("y")]

    # Extract probabilities
    # prob_cols = [i for i in results_df.columns if i.startswith("p")]
    # prob_cols = [i for i in prob_cols if len(i.split("_"))==2 and int(i.split("_")[1])-int(i.split("_")[0][1:]) == 1]

    results_df_x = results_df[x_pos_cols].astype(np.float32) + x_stride
    results_df_y = results_df[y_pos_cols].astype(np.float32) + y_stride
    # results_df_p = results_df[prob_cols].astype(np.float32)


    if cfg.frame_range is not None:
        # Extract the relevant frames
        frames = range(last_frame-first_frame+1)
        results_df_x = results_df_x.iloc[:, frames]
        results_df_y = results_df_y.iloc[:, frames]
        # results_df_p = results_df_p.iloc[:, range(cfg.frame_range[0], cfg.frame_range[1])]

    trajectory_beginning = np.zeros((results_df_x.shape[0], 2))

    for index, row in results_df_x.iterrows():
        trajectory_begin_idx = np.argmax(np.logical_not(np.isnan(row.to_numpy())) * 1)
        trajectory_beginning[index, 0] = row.iloc[trajectory_begin_idx]
        trajectory_beginning[index, 1] = results_df_y.iloc[index, trajectory_begin_idx]

    # Store lines for later manipulation
    all_lines = plt.plot(results_df_x.T, results_df_y.T, color="C0", marker=".", linewidth=1, picker=True, pickradius=5)

    # Give all lines information about which trajectory they are
    for idx, line in enumerate(all_lines):
        plt.setp(line, gid=str(idx))

    # Load the image and plot each frame
    if cfg.frame_range is not None:
        image = get_image_cut_as_ndarray(None, ["BF"], image_path, 
                                    upper_left_corner=(0, 0),
                                    pixel_dimensions=(-1,-1),
                                    frames=frames)
    else:
        image = get_image_cut_as_ndarray(None, ["BF"], image_path,
                                    upper_left_corner=(0, 0),
                                    pixel_dimensions=(-1,-1))
    
    if cfg.frame_range is not None:
        frames = range(cfg.frame_range[0], cfg.frame_range[1] + 1)
    else:
        frames = range(len(image))
    

    frames_imgs = []
    for i, frame in enumerate(image):
        f = plt.imshow(frame[0], cmap="gray", alpha=0.3)
        # By default, only have the first and last frames be visible
        if i != 0 or i != (len(image) - 1):
            f.set_visible(False)
        frames_imgs.append(f)


    # Define callback for clicking on a line (or other thing)
    def pickline(event):
        # Muste declare to use the global variables
        global cut_mode
        global swap_mode
        global selected_for_swapping
        global selected_for_swapping_frames
        global auxiliary_swap_lines_drawn
        global auxiliary_swap_lines
        global selected_for_cutting_trajectory
        global selected_for_cutting_frame
        global auxiliary_cut_lines_drawn
        # If we clicked a line
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()

            # Get trajectory id and frame
            prev_frame = int(min(event.ind))
            traj_id = int(plt.getp(thisline, "gid"))

            # Check that the line is well defined
            if prev_frame + 1 >= xdata.size or math.isnan(xdata[prev_frame + 1]):
                return

            if swap_mode:
                # If we clicked on teh same line twice, do nothing because we cant swap tracking within one trajectory
                if (len(selected_for_swapping) > 0 and selected_for_swapping[-1] == traj_id):
                    print("Invalid selection. Latest selected line is part of the same trajectory as the currently selected one.")
                    return

                # Datastructures to memorize the last two valid lines we clicked
                auxiliary_swap_lines.append(thisline)
                selected_for_swapping.append(traj_id)
                selected_for_swapping_frames.append(prev_frame)
                auxiliary_swap_lines_drawn.append(
                    plt.plot(xdata[[prev_frame, prev_frame + 1]], ydata[[prev_frame, prev_frame + 1]], color="lime",
                             marker=".", linewidth=1))

                # If we have more than two line selected, discard the oldest selected line
                if (len(selected_for_swapping) > 2):
                    auxiliary_swap_lines_drawn[0][0].remove()
                    auxiliary_swap_lines_drawn = auxiliary_swap_lines_drawn[-2:]
                    auxiliary_swap_lines = auxiliary_swap_lines[-2:]
                    selected_for_swapping = selected_for_swapping[-2:]
                    selected_for_swapping_frames = selected_for_swapping_frames[-2:]
                print(f"Trajectories registered for swapping: {selected_for_swapping} in frames {selected_for_swapping_frames}")

                # We only allow swapping tracking between the same frames (since we are sure about in which frame what happened).
                # Hence, if the operator selects two lines that are in different frames, adjust such that we select two lines from the trajectories selected, 
                # which are both in the last frame selected by the operator
                if (len(selected_for_swapping) == 2) and (selected_for_swapping_frames[0] != selected_for_swapping_frames[1]):
                    print("Line segments selected do not belong to same frame. Frame from last click gets enforced.")
                    auxiliary_swap_lines_drawn[0][0].remove()

                    # Check that the other trajectory even exists at the same time
                    if np.isnan(auxiliary_swap_lines[0].get_xdata()[[prev_frame, prev_frame + 1]].to_numpy()).any():
                        # We have a problem because the other selected trajectory does not exist at the same time
                        # We solve this problem by removing this incompatible trajectory
                        print("Frame enforcing not possible. Selections do not overlap in time.")
                        auxiliary_swap_lines_drawn = auxiliary_swap_lines_drawn[-1:]
                        auxiliary_swap_lines = auxiliary_swap_lines[-1:]
                        selected_for_swapping = selected_for_swapping[-1:]
                        selected_for_swapping_frames = selected_for_swapping_frames[-1:]
                    else:
                        auxiliary_swap_lines_drawn[0] = plt.plot(
                            auxiliary_swap_lines[0].get_xdata()[[prev_frame, prev_frame + 1]],
                            auxiliary_swap_lines[0].get_ydata()[[prev_frame, prev_frame + 1]], color="lime",
                            marker=".", linewidth=1)
                        selected_for_swapping_frames[0] = selected_for_swapping_frames[1]
                        print(f"Trajectories registered for swapping: {selected_for_swapping} in frames {selected_for_swapping_frames}")

                # If we have two lines selected, we are ready to swap
                if (len(selected_for_swapping) == 2):
                    print("Ready to swap. Press enter to confirm swap.")
                plt.gcf().canvas.draw_idle()
            if cut_mode:
                # If we already have a line selected, de-select that one
                if selected_for_cutting_trajectory is not None:
                    auxiliary_cut_lines_drawn[0].remove()

                # Select new line
                selected_for_cutting_trajectory = traj_id
                selected_for_cutting_frame = prev_frame
                auxiliary_cut_lines_drawn = plt.plot(xdata[[prev_frame, prev_frame + 1]],
                                                     ydata[[prev_frame, prev_frame + 1]], color="r", marker=".",
                                                     linewidth=1)
                print(f"Trajectory registered for cutting: {selected_for_cutting_trajectory} in frame {selected_for_cutting_frame}")
                print("Ready to cut. Press enter to confirm cut.")

    # Define a callback function that will be called when the user
    # finishes drawing the lasso on the image
    def onselect(verts):
        global all_lines
        global results_df_x
        global results_df
        global trajectory_beginning
        # The "verts" argument contains the coordinates of the
        # points that the user selected with the lasso
        path = matplotlibPath(verts)

        # Check if trajectory beginnings are within selected region
        inside = path.contains_points(trajectory_beginning)
        inside = pd.Series(inside, index=results_df.index)

        results_df.loc[inside, 'discard'] = ~results_df.loc[inside, 'discard']

        # Set the color of the selected points to orange
        for idx, value in results_df['discard'].items():
            all_lines[idx].set_color('C1' if not value else 'C0')

        # Redraw the figure
        plt.gcf().canvas.draw_idle()

    # Define a callback function that will be called when the user
    # presses a key
    def onpress(event):
        global swap_mode
        global results_df_x
        global results_df_y
        global results_df
        global all_lines
        global selected_for_swapping
        global selected_for_swapping_frames
        global auxiliary_swap_lines_drawn
        global auxiliary_swap_lines
        global cut_mode
        global selected_for_cutting_trajectory
        global selected_for_cutting_frame
        global auxiliary_cut_lines_drawn
        global trajectory_beginning
        print(f" > Pressed {event.key} < ")
        if event.key == SAVE_KEY:
            # Get the current date and time
            now = datetime.now()
            # Format the date and time to create a unique filename
            save_path = Path(RESULT_PATH / f"results_{image_name}_{now.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
            print(f"Saving new csv at {save_path}...")
            tmp_results_df = results_df.loc[:, results_df.columns != 'discard']
            tmp_results_df[results_df['discard'] == False].to_csv(save_path, index=False)
        if event.key == SWAPPING_KEY:
            # Toggle swap mode
            swap_mode = not swap_mode
            if swap_mode:
                cut_mode = False
                print("Swap mode enabled")
            else:
                print("Swap mode disabled")
        if event.key == CUTTING_KEY:
            # Toggle cutting mode
            cut_mode = not cut_mode
            if cut_mode:
                swap_mode = False
                print("Cut mode enabled")
            else:
                print("Cut mode disabled")

        if event.key in [str(i) for i in range(len(frames_imgs))]:
            frames_imgs[int(event.key)].set_visible(not frames_imgs[int(event.key)].get_visible())

            # Redraw the figure
            plt.gcf().canvas.draw_idle()

        # Confirm swapping (or other future tool)
        if event.key == CONFIRMATION_KEY:
            # If swap tool is currently selected
            if swap_mode:
                # If we are able to swap
                if len(selected_for_swapping) == 2:
                    print("Swapping...")
                    # Swap lines in results_df and then recompute results_df_x and results_df_y
                    tmp = results_df.iloc[selected_for_swapping[0],
                          1 + 2 * (selected_for_swapping_frames[0] + 1): -1].copy()
                    results_df.iloc[selected_for_swapping[0],
                    1 + 2 * (selected_for_swapping_frames[0] + 1): -1] = results_df.iloc[selected_for_swapping[1],
                                                                         1 + 2 * (selected_for_swapping_frames[
                                                                                      0] + 1): -1].copy()
                    results_df.iloc[selected_for_swapping[1], 1 + 2 * (selected_for_swapping_frames[0] + 1): -1] = tmp
                    results_df_x = results_df[[i for i in results_df.columns if str(i).startswith("x")]].astype(
                        np.float32)
                    results_df_y = results_df[[i for i in results_df.columns if str(i).startswith("y")]].astype(
                        np.float32)

                    # Adjust the drawings by updating teh relevant data
                    for swapped_idx in selected_for_swapping:
                        all_lines[swapped_idx].set_xdata(results_df_x.iloc[swapped_idx, :].to_numpy().T)
                        all_lines[swapped_idx].set_ydata(results_df_y.iloc[swapped_idx, :].to_numpy().T)
                    # Toggle off Swap mode for safety and clean up swapping datastructures
                    swap_mode = False
                    print("Swap mode toggled off for safety")
                    selected_for_swapping = []
                    selected_for_swapping_frames = []
                    for l in auxiliary_swap_lines_drawn:
                        l[0].remove()
                    auxiliary_swap_lines_drawn = []
                    auxiliary_swap_lines = []
                    plt.gcf().canvas.draw()
                    plt.gcf().canvas.flush_events()
                else:
                    print("Unable to swap. Missing selections.")
            if cut_mode:
                if selected_for_cutting_trajectory is None:
                    print("Unable to cut. Missing selection.")
                    return
                print("Cutting...")
                # Cut lines in results_df and then recompute results_df_x and results_df_y
                tmp = results_df.iloc[selected_for_cutting_trajectory, :].copy()
                tmp[1: 1 + 2 * (selected_for_cutting_frame + 1)] = pd.NA
                results_df.iloc[selected_for_cutting_trajectory, 1 + 2 * (selected_for_cutting_frame + 1): -1] = pd.NA
                results_df.loc[len(results_df.index)] = tmp
                results_df.iloc[-1, 0] = results_df.iloc[-2, 0] + 1
                results_df_x = results_df[[i for i in results_df.columns if str(i).startswith("x")]].astype(np.float32)
                results_df_y = results_df[[i for i in results_df.columns if str(i).startswith("y")]].astype(np.float32)

                # Update trajectory beginnings
                begin = np.asarray([results_df_x.iloc[-1, selected_for_cutting_frame + 1],
                                    results_df_y.iloc[-1, selected_for_cutting_frame + 1]])
                trajectory_beginning = np.vstack([trajectory_beginning, begin])

                # Adjust the drawings by updating teh relevant data and drawing a new line
                all_lines[selected_for_cutting_trajectory].set_xdata(
                    results_df_x.iloc[selected_for_cutting_trajectory, :].to_numpy().T)
                all_lines[selected_for_cutting_trajectory].set_ydata(
                    results_df_y.iloc[selected_for_cutting_trajectory, :].to_numpy().T)
                new_line = plt.plot(results_df_x.iloc[-1, :], results_df_y.iloc[-1, :], color="C0", marker=".",linewidth=1)
                plt.setp(new_line[0], gid=str(results_df.iloc[-1, 0]))

                all_lines = all_lines + new_line
                cut_mode = False
                print("Cut mode toggled off for safety")
                auxiliary_cut_lines_drawn[0].remove()
                selected_for_cutting_trajectory = None
                selected_for_cutting_frame = None
                auxiliary_cut_lines_drawn = None
                plt.gcf().canvas.draw()
                plt.gcf().canvas.flush_events()

    # Create the lasso selector and connect it to the image
    selector = LassoSelector(plt.gca(), onselect)

    # Listen for keypress events
    plt.gcf().canvas.mpl_connect('key_press_event', onpress)
    plt.gcf().canvas.mpl_connect('pick_event', pickline)

    # Show the plot
    print("---------- VISUALIZER INSTRUCTIONS ----------")
    print("Click >left mouse< and drag to select regions of droplets to keep. Selection is based on the initial location of the droplet.")
    print(f"Press >{SAVE_KEY}< to save an updated results csv which only contains droplets selected (marked with orange)")
    print(f"Press >{SWAPPING_KEY}< to toggle swap mode which can swap two edges")
    print(f"In swap mode, >left mouse< click on two edges and then press >{CONFIRMATION_KEY}< to confirm the edge swap")
    print(f"Press >{CUTTING_KEY}< to toggle cut mode which can cut a trajectory in two")
    print(f"In cut mode, >left mouse< click on an edge and then press >{CONFIRMATION_KEY}< to confirm the edge cut")
    for i in range(len(frames_imgs)):
        print(f"Press >{i}< to toggle the visibility of frame #{i}")
    plt.gcf().canvas.manager.set_window_title('Visualizer ' + image_name)
    plt.show()
