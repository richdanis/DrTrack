import pandas as pd
import numpy as np

# Used with paired_path = "c334_unstimulated_crop4x4.npy"
def generate_paired_patches(paired_path, data_path):
    
    patches = np.load("c334_unstimulated_crop4x4.npy")

    num_droplets = patches.shape[0]
    num_frames = patches.shape[1]
    channel_to_extract = 0

    droplet_patches = {'droplet_id': [], 'frame': [], 'patch': []}

    for i in range(num_frames):
        for j in range(num_droplets):
            droplet_patches["droplet_id"].append(j)
            droplet_patches["frame"].append(i)
            droplet_patches["patch"].append(patches[j, i, channel_to_extract, :, :])
        
    # And then save with np.save("paired_patches.npy", droplet_patches, allow_pickle=True)

    return droplet_patches