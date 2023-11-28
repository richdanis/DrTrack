import os
import time
import argparse
from pathlib import Path

from preprocess.for_detection import raw_to_preprocessed_for_detection
from preprocess.for_embeddings import raw_to_preprocessed_for_embeddings
from utils.globals import *

# import hydra
# from omegaconf import DictConfig, OmegaConf


def preprocess(raw_image_path: Path, image_name: str, pixel=-1):
    """Preprocess the .nd2 images and save as .npy arrays."""
    raw_to_preprocessed_for_detection(raw_image_path, image_name, PREPROCESSED_PATH, pixel=pixel)
    raw_to_preprocessed_for_embeddings(raw_image_path, image_name, PREPROCESSED_PATH, pixel=pixel)

#@hydra.main(config_path="conf", config_name="config", version_base=None)
def main():
    start_time = time.time()

    # TODO: Change the argument setup to hydra

    parser = argparse.ArgumentParser(description='Generate a tracking of droplets from an image.')

    parser.add_argument('raw_image', type=str,
                        help=f'ND2 image file name')

    parser.add_argument('-g', '--generate_embeddings', action='store_false',
                        help=f'generate embeddings. '
                             'default = True.')

    parser.add_argument('-t', '--train_model', action='store_true',
                        help=f'train model. '
                             'default = False')

    parser.add_argument('-s', '--skip_dataset', action='store_true',
                        help=f'Skip dataset generation. '
                             'default = False')

    parser.add_argument('-e', '--embeddings', action='store_false',
                        help=f'Use embeddings. '
                             'default = True')

    parser.add_argument('-l', '--linking', action='store_false',
                        help=f'Make linking. '
                             'default = True')

    parser.add_argument('--similarity_weight', type=float, default=0.5,
                        help='a number between 0 and 1. 1 means we only link droplets if they have exactly the same features. '
                             '0 means we allow linking of droplets even if they look very different. '
                             'default = 0.5')

    parser.add_argument('--vicinity_weight', type=float, default=0.5,
                        help='a number between 0 and 1. 1 means we prefer only spacial vicinity. '
                             '0 means we consider spacial vicinity and visual similarity. '
                             'default = 0.5')

    parser.add_argument('--max_dist', type=int, default=250,
                        help='a positive integer that indicates the maximal distance in pixels that a droplet can move between frames. '
                             'default = 250')

    parser.add_argument('--movement_variability', type=float, default=1.0,
                        help='a positive floating point number (typically close to 1, seem to work best). '
                             'Close to 0 means we prefer few droplets that move a lot. '
                             '1 means we are neutral. Greater than 1 means we prefer if many droplets move a bit. '
                             'default = 1')

    parser.add_argument('--radius_min', type=int, default=12,
                        help='Minimum radius of droplets to be detected in pixels. '
                             'default = 1')

    parser.add_argument('--radius_max', type=int, default=25,
                        help='Maximum radius of droplets to be detected in pixels. '
                             'default = 25')

    parser.add_argument('--pixel', type=int, default=-1,
                        help='number of pixels used per axis - just for training.')

    parser.add_argument('--skip_preprocessing', type=bool, default=False,
                        help='Whether to skip preprocessing. Defaults to False.')

    args = parser.parse_args()

    raw_image_path = Path(RAW_PATH / args.raw_image)
    image_name = args.raw_image[:-4].lower().replace(' ', '_')

    if args.pixel > 0:
        image_name = image_name + f'_{args.pixel}'

    if not args.skip_preprocessing:
        print("----Preprocess the Dataset for Detection and Embedding Creation----")
        preprocess(raw_image_path, image_name, pixel=args.pixel)
    else:
        print("----Skipping Preprocessing the Dataset----")


if __name__ == '__main__':
    main()
