import argparse
import sys

sys.path.append('src/')
sys.path.append('src/training/')
from training.models.efficientnet import EfficientNet


def load_model(args: argparse.Namespace):
    # TODO: needs to be changed if we want to be able to load different models.
    # Currently using EfficientNet, version b1, lightweight: 6.5 million parameters.
    return EfficientNet(args)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_data_path', type=str, default=None, help='Full path to .npy file with data.')
    parser.add_argument('--test_metadata_path', type=str, default=None, help='Full path to .npy file with data.')
    parser.add_argument('--model', default=None, help='Which model to use. Default is None.')
    parser.add_argument('--topk_accuracy', type=int, nargs='+', help='Whether to log validation topk accuracy.')
    parser.add_argument('--auroc_mode', type=str, default='roll',
                        help='Mode for calculating the negative class for AUROC')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Where model checkpoint is stored.')
    parser.add_argument('--validation_batch_size', type=int, default=32, help='Batch size. Default is 32.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use. Default is cpu.')
    parser.add_argument('--embed_dim', type=int, help='Dimension of the embedding layer.', required=True)
    parser.add_argument('--use_dapi', action='store_true', help='Whether to use DAPI channel.', default=False)

    return parser.parse_args()
