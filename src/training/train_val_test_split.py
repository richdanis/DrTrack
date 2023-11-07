import numpy as np
import torch
import os

RAW_CELL_DATASET_PATH = "data/cell_datasets/raw/"
CELL_DATASET_PATH = "data/cell_datasets/"

def add_labels_and_split(dataset):
        
        # turn to torch for flatten function
        dataset = torch.from_numpy(dataset)
        
        # labels are given by droplets in the following frame
        labels = dataset[:,1:,:,:,:]
        
        # remove last frame
        x = dataset[:,:-1,:,:,:]

        # split into 80/10/10 train, validation, and test sets
        x_train = x[:int(len(x)*0.8)]
        labels_train = labels[:int(len(labels)*0.8)]

        x_validation = x[int(len(x)*0.8):int(len(x)*0.9)]
        labels_validation = labels[int(len(labels)*0.8):int(len(labels)*0.9)]

        x_test = x[int(len(x)*0.9):]
        labels_test = labels[int(len(labels)*0.9):]

        # flatten both along frames and concatenate
        train = flatten_and_concatenate(x_train, labels_train)

        validation = flatten_and_concatenate(x_validation, labels_validation)

        test = flatten_and_concatenate(x_test, labels_test)
        
        return train.numpy(), validation.numpy(), test.numpy()

def flatten_and_concatenate(x, y):
    
    # flatten both along frames and concatenate
    y = torch.flatten(y, start_dim=0, \
                                    end_dim=1)
        
    x = torch.flatten(x, start_dim=0, \
                               end_dim=1)
    
    # insert new dimension at position 1 (for concatenation)
    y = torch.unsqueeze(y, 1)
    x = torch.unsqueeze(x, 1)
    
    return torch.cat((x, y), dim=1)

# TODO: Think about how to sample from each raw cell datasets,
# as they have very different sizes.

def main():
    """
    Split the raw cell datasets into train, validation, and test sets.
    Output: train.npy, validation.npy, test.npy
    Format: (N, 2, C, H, W)
    N: number of patches
    Second dimension: input and label (in this order)
    C: number of channels
    H: height
    W: width
    """
    train, validation, test = None, None, None

    # load cell datasets and concatenate them
    for fname in os.listdir(RAW_CELL_DATASET_PATH):

        dataset = np.load(RAW_CELL_DATASET_PATH + fname)


        if train is None:
            train, validation, test = add_labels_and_split(dataset)
        else:
            train_, validation_, test_ = add_labels_and_split(dataset)
            train = np.concatenate((train, train_), axis=0)
            validation = np.concatenate((validation, validation_), axis=0)
            test = np.concatenate((test, test_), axis=0)

    print("Train set size: " + str(len(train)))
    print("Validation set size: " + str(len(validation)))
    print("Test set size: " + str(len(test)))

    print("Shape: " + str(train.shape))

    # fix seed for reproducibility
    np.random.seed(0)

    # shuffle the datasets
    np.random.shuffle(train)
    np.random.shuffle(validation)
    np.random.shuffle(test)

    # save the datasets
    np.save(CELL_DATASET_PATH + "train.npy", train)
    np.save(CELL_DATASET_PATH + "validation.npy", validation)
    np.save(CELL_DATASET_PATH + "test.npy", test)

if __name__ == "__main__":
    main()