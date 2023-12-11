from pathlib import Path

import hydra
from omegaconf import DictConfig

from visualizer import interactive_explorer

@hydra.main(config_path="../conf", config_name="visualizer", version_base=None)
def main(cfg: DictConfig):

    assert cfg.raw_image[-4:] == '.nd2', 'Image mast be an .nd2 file, or add ".nd2" to the end of the name'
    assert cfg.results[-4:] == '.csv', 'Results mast be an .csv file, or add ".csv" to the end of the name'

    image_name = cfg.raw_image[:-4].lower().replace(' ', '_')
    raw_image_path = Path(f"{cfg.data_path}/{cfg.raw_dir}/{cfg.raw_image}")
    results_path = Path(f"{cfg.data_path}/{cfg.results_dir}/{cfg.experiment_name}/{cfg.results}")

    interactive_explorer.select_trajectories(raw_image_path, results_path, image_name, cfg.pixel)


if __name__ == "__main__":
    main()
