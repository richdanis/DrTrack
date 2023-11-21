import numpy as np
import tqdm
import os
import argparse

DATA_PATH = '/cluster/scratch/rdanis/data/local_datasets/'

def main():

    # read in files to merge
    parser = argparse.ArgumentParser()

    parser.add_argument('--fnames', nargs='+', type=str, required=True, help='List of folders to merge')
    parser.add_argument('--outname', type=str, required=True, help='Name of output folder')

    args = parser.parse_args()

    # read in folders
    labels, negatives, patches = None, None, None

    for fname in tqdm.tqdm(args.fnames, desc='Merging'):
        
        path = os.path.join(DATA_PATH, fname)

        if labels is None:
            labels = np.load(os.path.join(path, 'labels.npy'))
            negatives = np.load(os.path.join(path, 'negatives.npy'))
            patches = np.load(os.path.join(path, 'patches.npy'))

            if len(labels) < 2000:
                labels, negatives, patches = None, None, None

        else:
            labels_new = np.load(os.path.join(path, 'labels.npy'))
            negatives_new = np.load(os.path.join(path, 'negatives.npy'))
            patches_new = np.load(os.path.join(path, 'patches.npy'))

            if len(labels_new) < 2000:
                continue

            # merging logic
            last_patches1 = patches[len(labels):]
            last_patches2 = patches_new[len(labels_new):]

            patches = np.concatenate((patches[:len(labels)], patches_new[:len(labels_new)]), axis=0)

            # for ease of label shifting lastpatches2 ist first added
            patches = np.concatenate((patches, last_patches2, last_patches1), axis=0)

            # shift new indices by len(labels)
            negatives_new += len(labels)
            labels_new += len(labels)

            # shift old indices when necessary
            for i in range(len(labels)):
                if labels[i] >= len(labels):
                    labels[i] += len(patches_new)
                for j in range(len(negatives[i])):
                    if negatives[i][j] >= len(labels):
                        negatives[i][j] += len(patches_new)

            # concatenate labels and negatives
            labels = np.concatenate((labels, labels_new), axis=0)
            negatives = np.concatenate((negatives, negatives_new), axis=0)

            # TODO: permutation
    
    # save merged dataset
    path = os.path.join(DATA_PATH, args.outname)

    if not os.path.exists(path):
        os.makedirs(path)

    print("Amount of patches: ", len(patches))

    np.save(os.path.join(path, 'labels.npy'), labels)
    np.save(os.path.join(path, 'negatives.npy'), negatives)
    np.save(os.path.join(path, 'patches.npy'), patches)

# TODO: verify that this works


if __name__ == '__main__':
    main()
