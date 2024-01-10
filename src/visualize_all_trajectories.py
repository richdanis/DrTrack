# Types
from pathlib import Path
from omegaconf import DictConfig

# Configuration
import hydra

# String manipulation
import re

# Local imports
from visualizer import interactive_explorer

def get_stride(results_str: str) -> tuple:
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


@hydra.main(config_path="../conf", config_name="config_visualize_all_trajectories", version_base=None)
def main(cfg: DictConfig):
    # Check that the image and results are in the correct format.
    assert cfg.raw_image[-4:] == '.nd2', 'Image mast be an .nd2 file, or add ".nd2" to the end of the name'
    assert cfg.results[-4:] == '.csv', 'Results mast be an .csv file, or add ".csv" to the end of the name'

    # Get paths
    image_name = cfg.raw_image[:-4].lower().replace(' ', '_')
    raw_image_path = Path(f"{cfg.data_path}/{cfg.raw_dir}/{cfg.raw_image}")
    results_path = Path(f"{cfg.data_path}/{cfg.results_dir}/{cfg.experiment_name}/{cfg.results}")

    # Get the stride
    if cfg.evaluation_results:
        y_stride = 0
        x_stride = 0
    else:
        y_stride, x_stride = get_stride(cfg.results[:-4])

    # Run the interactive explorer
    interactive_explorer.launch_interactive_explorer(cfg, raw_image_path, results_path, image_name, y_stride, x_stride)

if __name__ == "__main__":
    main()
