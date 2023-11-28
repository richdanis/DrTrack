import os
import time
import argparse
from pathlib import Path
import nd2

from preprocess.for_detection import raw_to_preprocessed_for_detection
from preprocess.for_embeddings import raw_to_preprocessed_for_embeddings
from preprocess.for_detection import raw_cut_to_preprocessed_for_detection
from preprocess.for_embeddings import raw_cut_to_preprocessed_for_embeddings
from preprocess.for_all import preprocess_all_cuts_and_store


from utils.globals import *

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
     # Start timer
     start_time = time.time()

     # Preprocess the image cuts if required
     # Check conf/preprocess.yaml for settings
     image_name = cfg.preprocess.raw_image[:-4].lower().replace(' ', '_')
     preprocessed_path_img = Path(PREPROCESSED_PATH / image_name)

     if not cfg.preprocess.skip:
         # Create paths if they do not exist
          if not os.path.exists(preprocessed_path_img): 
               os.makedirs(preprocessed_path_img) 

          cut_names = preprocess_all_cuts_and_store(cfg, RAW_PATH, preprocessed_path_img, image_name)
   
if __name__ == '__main__':
    main()
