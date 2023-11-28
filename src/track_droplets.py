import os
import time
import argparse
from pathlib import Path

from preprocess.for_detection import raw_to_preprocessed_for_detection
from preprocess.for_embeddings import raw_to_preprocessed_for_embeddings
from utils.globals import *

import hydra
from omegaconf import DictConfig, OmegaConf

# Run: python track_droplets.py "Small mvt 1.nd2" --pixel 100 -e

def preprocess(cfg, raw_image_path: Path, image_name: str, pixel=-1):
    """Preprocess the .nd2 images and save as .npy arrays."""
    raw_to_preprocessed_for_detection(cfg, raw_image_path, image_name, PREPROCESSED_PATH, pixel=pixel)
    raw_to_preprocessed_for_embeddings(cfg, raw_image_path, image_name, PREPROCESSED_PATH, pixel=pixel)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
     start_time = time.time()

     raw_image_path = Path(RAW_PATH / cfg.preprocess.raw_image)
     image_name = cfg.preprocess.raw_image[:-4].lower().replace(' ', '_')

     if cfg.preprocess.pixel > 0:
          image_name = image_name + f'_{cfg.preprocess.pixel}'

     if not cfg.preprocess.skip:
          print("----Preprocess the Dataset for Detection and Embedding Creation----")
          preprocess(cfg, raw_image_path, image_name, pixel=cfg.preprocess.pixel)
     else:
          print("----Skipping Preprocessing the Dataset----")


if __name__ == '__main__':
    main()
