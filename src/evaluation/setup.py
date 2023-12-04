import argparse
import sys

sys.path.append('src/')
sys.path.append('src/training/')

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_data_path', type=str, default=None, help='Full path to .npy file with data.')
    parser.add_argument('--test_metadata_path', type=str, default=None, help='Full path to .npy file with data.')
    parser.add_argument('--model', default=None, help='Which model to use. Default is None.')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Where model checkpoint is stored.')
    parser.add_argument('--validation_batch_size', type=int, default=32, help='Batch size. Default is 32.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use. Default is cpu.')
    parser.add_argument('--embed_dim', type=int, help='Dimension of the embedding layer.', required=True)
    parser.add_argument('--use_dapi', action='store_true', help='Whether to use DAPI channel.', default=False)

    return parser.parse_args()
