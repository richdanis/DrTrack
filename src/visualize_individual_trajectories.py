import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import pandas as pd
import matplotlib.widgets as widgets
from pathlib import Path
import os
import re
from preprocess.raw_image_reader import get_image_cut_as_ndarray
import hydra
from omegaconf import DictConfig


class Trajectories:
    """
    Class to handle the trajectories from the tracking algorithm.

    Parameters
    ----------
    cfg : DictConfig
        The configuration file.
    RAW_PATH : Path
        The path to the raw image.
    RESULTS_PATH : Path
        The path to the results file.

    Attributes
    ----------
    final_output : pd.DataFrame
        The final output of the tracking algorithm.
    frames : list
        The frames to be visualized.
    channels : list
        The channels to be visualized.
    image : np.ndarray
        The image to be visualized.
    trajectories : pd.DataFrame
        The trajectories to be visualized.
    radius : int
        The radius of the patch.
    y_max : int
        The maximum y value.
    x_max : int
        The maximum x value.

    Methods
    -------
    get_stride(results_str: str) -> tuple
        Get the correct position of the droplet by adding the stride.
    adjust_positions(df: pd.DataFrame, y_stride: int, x_stride: int) -> pd.DataFrame
        Get x, y positions of the droplets in the given frames.
    get_patch(frame, center_y: int, center_x: int)
        Get the patch of the droplet.
    get_group_patch(row) -> np.ndarray
        Get patches.
    """

    def __init__(self, cfg, RAW_PATH, RESULTS_PATH):

        if os.path.exists(f"{RESULTS_PATH}/{cfg.experiment_name}/flagged_{cfg.results}"):
            tmp = pd.read_csv(f"{RESULTS_PATH}/{cfg.experiment_name}/flagged_{cfg.results}")
        else:
            tmp = pd.read_csv(f"{RESULTS_PATH}/{cfg.experiment_name}/{cfg.results}")

        self.final_output = tmp
        counter = 0
        for col in tmp.columns:
            if col.startswith('id'):
                counter += 1

        end_frame = counter if cfg.end_frame == -1 else cfg.end_frame
        self.frames = list(range(cfg.start_frame, end_frame))

        self.channels = ['DAPI', 'BF'] if not cfg.all_channels else ['DAPI', 'FITC', 'TRITC', 'Cy5', 'BF']
        IMAGE_PATH = Path(RAW_PATH / cfg.image_name)

        self.image = get_image_cut_as_ndarray(None, channels=self.channels, path_to_image=IMAGE_PATH,
                                              upper_left_corner=(0, 0),
                                              pixel_dimensions=(-1, -1),
                                              frames=self.frames,
                                              pixel=cfg.pixel)

        # cut all columns that are not needed since frames are specified
        full_prob_col = self.final_output.columns[
            self.final_output.columns.str.match(f'p{cfg.start_frame}-{end_frame - 1}')]
        full_traj_prob = self.final_output[full_prob_col]
        tmp = self.final_output.drop(
            columns=self.final_output.columns[self.final_output.columns.str.match('p(\d+)-(\d+)')])
        tmp = tmp.drop(columns=['full_trajectory_uncertainty'])
        tmp = tmp.drop(columns=self.final_output.columns[self.final_output.columns.str.startswith('id')])
        x_cols = tmp.columns[tmp.columns.str.startswith('x')]
        y_cols = tmp.columns[tmp.columns.str.startswith('y')]
        nr_cells_cols = tmp.columns[tmp.columns.str.startswith('nr_cells')]
        for col in x_cols:
            if int(col[1:]) not in self.frames:
                tmp = tmp.drop(columns=col)
        for col in y_cols:
            if int(col[1:]) not in self.frames:
                tmp = tmp.drop(columns=col)
        for col in nr_cells_cols:
            if int(col[8:]) not in self.frames:
                tmp = tmp.drop(columns=col)
        probs = tmp.columns[tmp.columns.str.match('p(\d+)_(\d+)')]
        pattern = r'p(\d+)_(\d+)'
        for col in probs:
            match = re.search(pattern, col)
            if (int(match.group(1)) not in self.frames) or (int(match.group(2)) not in self.frames):
                tmp = tmp.drop(columns=col)

        tmp['trajectory_uncertainty'] = full_traj_prob

        # build the trajectories dataframe with the correct columns according to the stride
        self.trajectories = tmp
        stride = self.get_stride(cfg.results)
        self.trajectories = self.adjust_positions(self.trajectories, stride[0], stride[1])

        self.radius = cfg.radius
        self.y_max = self.image.shape[2]
        self.x_max = self.image.shape[3]


    def get_stride(self, results_str: str) -> tuple:
        """
        Get the correct position of the droplet by adding the stride.

        Parameters
        ----------
        results_str : str
            The name of the results file.

        Returns
        -------
        tuple
            The y and x strides.
        """
        # Define a pattern using regular expression
        pattern = r'y(\d+)_x(\d+)'
        # Use re.search to find the pattern in the string
        match = re.search(pattern, results_str)
        # Extract the x and y values from the matched groups
        return int(match.group(1)), int(match.group(2))


    def adjust_positions(self, df: pd.DataFrame,
                         y_stride: int,
                         x_stride: int) -> pd.DataFrame:
        """
        Get x, y positions of the droplets in the given frames.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with the positions.
        y_stride : int
            The y stride.
        x_stride : int
            The x stride.

        Returns
        -------
        pd.DataFrame
            The adjusted dataframe.
        """ 
        # Add the stride to the x and y positions
        df[df.columns[df.columns.str.startswith('x')]] = df[df.columns[df.columns.str.startswith('x')]] + x_stride
        df[df.columns[df.columns.str.startswith('y')]] = df[df.columns[df.columns.str.startswith('y')]] + y_stride
        return df

    # Get the patch of the droplet
    def get_patch(self, frame, center_y: int, center_x: int):
        """
        Get the patch of the droplet.

        Parameters
        ----------
        frame : int
            The frame.
        center_y : int
            The y coordinate.
        center_x : int
            The x coordinate.

        Returns
        -------
        np.ndarray
            The image patch for the given droplet.
        """
        # We are in the case where we have channel, image_row and image_col as axes.
        window_y = np.asarray((min(max(0, center_y - self.radius), self.y_max - 1),
                               max(0, min(self.y_max, center_y + self.radius))),
                              dtype=np.int32)

        window_x = np.asarray((min(max(0, center_x - self.radius), self.x_max - 1),
                               max(0, min(self.x_max - 1, center_x + self.radius))),
                              dtype=np.int32)

        ans = self.image[frame, :, window_y[0]: window_y[1], window_x[0]: window_x[1]]
        if ans.shape[1] != 2 * self.radius or ans.shape[2] != 2 * self.radius:
            print("Warning: Droplet is too close to the edge of the image. The patch is padded with zeros.")
            tmp = np.zeros((2, 2 * self.radius, 2 * self.radius))
            tmp[:, :ans.shape[1], :ans.shape[2]] = ans
            ans = tmp
        return ans


    def get_group_patch(self, row: int) -> np.ndarray:
        """
        Get patches.
        
        Parameters
        ----------
        row : int
            The row numer.
        
        Returns
        -------
        np.ndarray
            The image patches for the given droplet.
        """
        traj = self.trajectories.iloc[row]
        # Get the x and y coordinates along with the trajectory
        x_pos = traj[traj.index.str.startswith('x')].values
        y_pos = traj[traj.index.str.startswith('y')].values
        # Get the patches
        patches = []
        for frame, (y, x) in enumerate(zip(y_pos, x_pos)):
            patches.append(self.get_patch(frame, y, x))
        return np.array(patches)


class Visualizer:
    """
    Class to visualize the trajectories.
    
    Parameters
    ----------
    cfg : DictConfig
        The configuration file.
    RAW_PATH : Path
        The path to the raw image.
    RESULTS_PATH : Path
        The path to the results file.

    Attributes
    ----------
    traj : Trajectories
        The trajectories.
    number_of_rows : int
        The number of rows.
    results_path : Path
        The path to the results file.
    all_channels : bool
        Whether to visualize all channels.
    verbose : bool
        Whether to print information.
    save : bool
        Whether to save the results.
    file : str
        The name of the results file.
    fig : plt.figure
        The figure.
    axarr : np.ndarray
        The axes.
    current_idx : int
        The current index.
    rows_false : list
        The rows marked as false.
    rows_true : list
        The rows marked as true.
    rows_unsure : list
        The rows marked as unsure.
    label_text : plt.text
        The label text.
    row_info_text : plt.text
        The row info text.
    
    Methods
    -------
    update_display()
        Update display for a new row.
    on_true_clicked(event)
        Mark the current row for keeping, and add label 'True'.
    on_unsure_clicked(event)
        Mark the current row for keeping, and add label 'Unsure'.
    on_false_clicked(event)
        Mark the current row for deletion, and add label 'False'.
    on_delete_clicked(event)
        If the current row is marked for deletion, unmark it.
    on_prev_clicked(event)
        Go to the previous row.
    on_next_clicked(event)
        Go to the next row.
    on_close(event)
        Label the rows in rows_false with "False", the rows in rows_true with "True", and the rows in rows_unsure with "Unsure".
    on_key(event)
        Keyboard interaction.
    on_submit(text_box, text)
        Jump to specified row when user submits a value in the TextBox.
    """

    def __init__(self, cfg, RAW_PATH, RESULTS_PATH):
        self.traj = Trajectories(cfg, RAW_PATH, RESULTS_PATH)
        self.number_of_rows = len(self.traj.final_output)
        self.results_path = RESULTS_PATH
        self.all_channels = cfg.all_channels
        self.verbose = cfg.verbose
        self.save = cfg.save
        self.file = cfg.results

        self.fig, self.axarr = plt.subplots(2, len(self.traj.frames), figsize=(15, 8))
        if self.all_channels:
            self.fig, self.axarr = plt.subplots(5, len(self.traj.frames), figsize=(15, 20))

        self.current_idx = 0

        # Create a list to keep track of rows
        self.rows_false = []
        self.rows_true = []
        self.rows_unsure = []

        # Remove axis for each subplot
        for ax_row in self.axarr:
            for ax in ax_row:
                ax.set_yticklabels([])
                ax.set_xticklabels([])

        # Add annotation to display the current label
        self.label_text = self.fig.text(0.5, 0.95, '', ha='center', transform=self.fig.transFigure)
        # Add annotation to display the current row info
        self.row_info_text = self.fig.text(0.15, 0.95, '', transform=self.fig.transFigure)

        # Define a TextBox widget for user input
        # Adjust position & size as needed
        ax_textbox = plt.axes([0.8, 0.925, 0.1, 0.05])  # [x, y , width, height]
        text_box = widgets.TextBox(ax_textbox, 'Jump to Row:')
        # Assuming text_box is created as mentioned in the previous response
        text_box.on_submit(lambda text: self.on_submit(text_box, text))

        # Display the first row
        self.update_display()  # start with the first row

        # Middle position of the figure's width
        middle = 0.5

        # Define button widths and offsets from the middle
        button_width = 0.1
        offset = 0.1
        button_height = 0.05
        button_space = 0.01  # Vertical space between buttons and plots

        # Compute bottom position based on button's height and space from plots
        bottom_position = button_space + button_height

        # Define the button axes uniformly distributed
        ax_prev = plt.axes([middle - 3 * offset, bottom_position, button_width, button_height])
        ax_true = plt.axes([middle - 2 * offset, bottom_position, button_width, button_height])
        ax_unsure = plt.axes([middle - offset, bottom_position, button_width, button_height])
        ax_false = plt.axes([middle, bottom_position, button_width, button_height])
        ax_delete = plt.axes([middle + 1 * offset, bottom_position, button_width, button_height])
        ax_next = plt.axes([middle + 2 * offset, bottom_position, button_width, button_height])

        # Create buttons
        btn_prev = Button(ax_prev, 'Prev')
        btn_true = Button(ax_true, 'True (1)')
        btn_unsure = Button(ax_unsure, 'Unsure (2)')
        btn_false = Button(ax_false, 'False (3)')
        btn_delete = Button(ax_delete, 'Delete (4)')
        btn_next = Button(ax_next, 'Next')

        # Link buttons to their callback functions
        btn_prev.on_clicked(self.on_prev_clicked)
        btn_true.on_clicked(self.on_true_clicked)
        btn_unsure.on_clicked(self.on_unsure_clicked)
        btn_false.on_clicked(self.on_false_clicked)
        btn_delete.on_clicked(self.on_delete_clicked)
        btn_next.on_clicked(self.on_next_clicked)

        # Link the key event to the callback function
        plt.gcf().canvas.mpl_connect('key_press_event', self.on_key)

        # Connect the close event to the on_close function
        plt.connect('close_event', self.on_close)
        plt.show()


    def update_display(self):
        """
        Update display for a new row.
        
        Returns
        -------
        None
        """
        images = self.traj.get_group_patch(self.current_idx)
        if self.all_channels:
            for col_idx, (img_0, img_1, img_2, img_3, img_4) in enumerate(images):
                # img_0 == DAPI, img_1 == FITC, img_2 == TRITC, img_3 == Cy5, img_4 == BF
                # But want to display in the order BF, DAPI, FITC, TRITC, Cy5
                self.axarr[0, col_idx].imshow(img_4, vmin=0)
                if col_idx == 0:
                    self.axarr[0, col_idx].set_title(
                        'Aggregated: ' + str(self.traj.trajectories.iloc[self.current_idx][f'trajectory_uncertainty']))
                else:
                    self.axarr[0, col_idx].set_title('Prob: ' + str(self.traj.trajectories.iloc[self.current_idx][
                                                                        f'p{self.traj.frames[col_idx - 1]}_{self.traj.frames[col_idx]}']))
                self.axarr[0, col_idx].set_xlabel(
                    'x: ' + str(int(self.traj.trajectories.iloc[self.current_idx][f'x{self.traj.frames[col_idx]}']))
                    + ' y: ' + str(int(self.traj.trajectories.iloc[self.current_idx][f'y{self.traj.frames[col_idx]}'])),
                    fontsize=10)
                self.axarr[1, col_idx].set_title('Cells: ' + str(
                    int(self.traj.trajectories.iloc[self.current_idx][f'nr_cells{self.traj.frames[col_idx]}'])))
                self.axarr[1, col_idx].imshow(img_0, vmin=0)
                self.axarr[2, col_idx].imshow(img_1, vmin=0)
                self.axarr[3, col_idx].imshow(img_2, vmin=0)
                self.axarr[4, col_idx].imshow(img_3, vmin=0)

            # the original order is DAPI, FITC, TRITC, Cy5, BF
            self.axarr[0, 0].set_ylabel("BF", fontsize=10)
            self.axarr[1, 0].set_ylabel("DAPI", fontsize=10)
            self.axarr[2, 0].set_ylabel("FITC", fontsize=10)
            self.axarr[3, 0].set_ylabel("TRITC", fontsize=10)
            self.axarr[4, 0].set_ylabel("Cy5", fontsize=10)


        else:
            for col_idx, (img_0, img_1) in enumerate(images):
                # img_0 == DAPI, img_1 == BF
                # But want to display in the order BF, DAPI
                self.axarr[0, col_idx].imshow(img_1, vmin=0)
                if col_idx == 0:
                    self.axarr[0, col_idx].set_title(
                        'Aggregated: ' + str(self.traj.trajectories.iloc[self.current_idx][f'trajectory_uncertainty']))
                else:
                    self.axarr[0, col_idx].set_title('Prob: ' + str(self.traj.trajectories.iloc[self.current_idx][
                                                                        f'p{self.traj.frames[col_idx - 1]}_{self.traj.frames[col_idx]}']))
                self.axarr[0, col_idx].set_xlabel(
                    'x: ' + str(int(self.traj.trajectories.iloc[self.current_idx][f'x{self.traj.frames[col_idx]}']))
                    + ' y: ' + str(int(self.traj.trajectories.iloc[self.current_idx][f'y{self.traj.frames[col_idx]}'])),
                    fontsize=10)
                self.axarr[1, col_idx].set_title('Cells: ' + str(
                    int(self.traj.trajectories.iloc[self.current_idx][f'nr_cells{self.traj.frames[col_idx]}'])))
                self.axarr[1, col_idx].imshow(img_0, vmin=0)

            # the original order is DAPI, BF
            self.axarr[0, 0].set_ylabel("BF", fontsize=10)
            self.axarr[1, 0].set_ylabel("DAPI", fontsize=10)

        plt.draw()
        # Update the label text
        self.row_info_text.set_text(f"Row: {self.current_idx + 1} / {self.number_of_rows}")
        # Update the row info text
        if self.current_idx in self.rows_false:
            self.label_text.set_text('False')
        elif self.current_idx in self.rows_true:
            self.label_text.set_text('True')
        elif self.current_idx in self.rows_unsure:
            self.label_text.set_text('Unsure')
        else:
            self.label_text.set_text('')

    def on_true_clicked(self, event):
        """
        Mark the current row for keeping, and add label 'True'.
        
        Parameters
        ----------
        event : event
            The event.
        """
        if self.verbose:
            print(f"Row {self.current_idx} is marked true.")
        # Mark the current row for keeping, and add label 'True'
        self.rows_true.append(self.traj.final_output.index[self.current_idx])
        tmp = self.traj.final_output.index[self.current_idx]
        if tmp in self.rows_false:
            self.rows_false.remove(tmp)
        elif tmp in self.rows_unsure:
            self.rows_unsure.remove(tmp)
        self.on_next_clicked(event)


    def on_unsure_clicked(self, event):
        """
        Mark the current row for keeping, and add label 'Unsure'.
        
        Parameters
        ----------
        event : event
            The event.
        """
        if self.verbose:
            print(f"Row {self.current_idx} is marked unsure.")
        # Mark the current row for keeping, and add label 'Unsure'
        self.rows_unsure.append(self.traj.final_output.index[self.current_idx])
        tmp = self.traj.final_output.index[self.current_idx]
        if tmp in self.rows_false:
            self.rows_false.remove(tmp)
        elif tmp in self.rows_true:
            self.rows_true.remove(tmp)
        self.on_next_clicked(event)


    def on_false_clicked(self, event):
        """
        Mark the current row for deletion, and add label 'False'.
        
        Parameters
        ----------
        event : event
            The event.
        """
        if self.verbose:
            print(f"Row {self.current_idx} is marked false.")
        # Mark the current row for deletion, and add label 'False'
        self.rows_false.append(self.traj.final_output.index[self.current_idx])
        tmp = self.traj.final_output.index[self.current_idx]
        if tmp in self.rows_true:
            self.rows_true.remove(tmp)
        elif tmp in self.rows_unsure:
            self.rows_unsure.remove(tmp)
        self.on_next_clicked(event)


    def on_delete_clicked(self, event):
        """
        If the current row is marked for deletion, unmark it.
        
        Parameters
        ----------
        event : event
            The event.
        """
        if self.verbose:
            print(f"Label of row {self.current_idx} has be removed.")
        # If the current row is marked for deletion, unmark it
        if self.current_idx in self.rows_false:
            self.rows_false.remove(self.current_idx)
        elif self.current_idx in self.rows_true:
            self.rows_true.remove(self.current_idx)
        elif self.current_idx in self.rows_unsure:
            self.rows_unsure.remove(self.current_idx)
        self.on_next_clicked(event)


    def on_prev_clicked(self, event):
        """
        Go to the previous row.
        
        Parameters
        ----------
        event : event
            The event.
        
        Returns
        -------
        None
        """
        if self.current_idx == 0:
            self.current_idx = self.number_of_rows - 1
        else:
            self.current_idx -= 1
        self.update_display()

    def on_next_clicked(self, event):
        """
        Go to the next row.
        
        Parameters
        ----------
        event : event
            The event.
            
        Returns
        -------
        None
        """
        if self.current_idx == self.number_of_rows - 1:
            self.current_idx = 0
        else:
            self.current_idx += 1
        self.update_display()

    def on_close(self, event):
        """
        Label the rows in rows_false with "False", the rows in rows_true with "True", and the rows in rows_unsure with "Unsure".

        Parameters
        ----------
        event : event
            The event.

        Returns
        -------
        None
        """
        # Label the rows in rows_false with "False", the rows in rows_true with "True", and the rows in rows_unsure with "Unsure"
        if 'Label' not in self.traj.final_output.columns:
            self.traj.final_output['Label'] = np.nan

        self.traj.final_output.loc[self.rows_false, 'Label'] = "False"
        self.traj.final_output.loc[self.rows_true, 'Label'] = "True"
        self.traj.final_output.loc[self.rows_unsure, 'Label'] = "Unsure"

        save_name = f"{self.results_path}/flagged_results_{self.file}.csv"

        # Save the refined_df to CSV
        if self.save:
            self.traj.final_output.to_csv(save_name, index=False)
            print('Saved the refined tracked droplet data after labeling.')
        else:
            print('The refined tracked droplet data is not saved.')

    def on_key(self, event):
        """
        Execute action depending on which key was pressed.
        
        Parameters
        ----------
        event : event
            The event.

        Returns
        -------
        None
        """

        if event.key == 'right':
            self.on_next_clicked(event)
        elif event.key == 'left':
            self.on_prev_clicked(event)
        elif event.key == '1':
            self.on_true_clicked(event)
        elif event.key == '2':
            self.on_unsure_clicked(event)
        elif event.key == '3':
            self.on_false_clicked(event)
        elif event.key == '4':
            self.on_delete_clicked(event)

    def on_submit(self, text_box, text):
        """
        Jump to specified row when user submits a value in the TextBox.

        Parameters
        ----------
        text_box : TextBox
            The TextBox widget.
        text : str
            The text submitted by the user.

        Returns
        -------
        None
        """
        # Check if text is not empty
        if text.strip():  # This removes any leading/trailing whitespace and checks if text is not just whitespace
            try:
                # Convert text input to integer and adjust for 0-based indexing
                new_idx = int(text) - 1
                # Check if the new index is within the dataframe's bounds
                if 0 <= new_idx < self.number_of_rows:
                    self.current_idx = new_idx
                    self.update_display()
                else:
                    print(f"Row {text} is out of range!")
            except ValueError:
                print("Please enter a valid row number.")
        # Clear the contents of the TextBox after processing the input
        text_box.set_val('')


@hydra.main(config_path="../conf", config_name="config_visualize_individual_trajectories", version_base=None)
def main(cfg: DictConfig):
    RAW_PATH = Path(cfg.data_path) / Path(cfg.raw_dir)
    RESULTS_PATH = Path(cfg.data_path) / Path(cfg.results_dir)

    # Run the visualizer
    Visualizer(cfg, RAW_PATH, RESULTS_PATH)

if __name__ == "__main__":
    main()
