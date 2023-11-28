import numpy as np
from . import cell_finding

# takes in the dapi channel, the bf channel and a list of detected droplets
# returns three matrices. The first is a mask that is 1 at a location where a significant peak in the dapi channel has been found
# The second is a matrix that contains some intensity measure of the peaks in the dapi channel.
# The third is a matrix that contains a persistency measure (basically some sort of width of the peaks) of the peaks found in the dapi channel.
def cell_detector(cfg, dapi_channel, bf_channel, detected_droplets, stats_print=False):

    cell_detection_scores = cell_finding.cell_finding(cfg, detected_droplets, dapi_channel, bf_channel)
    detected_cells_mask = cell_detection_scores[0, :, :]
    intensity_scores = cell_detection_scores[1, :, :]
    persistency_scores = cell_detection_scores[2, :, :]

    squashed_intensity = (1.0 - np.exp(-intensity_scores)) * detected_cells_mask
    squashed_persistency = (1.0 - np.exp(-persistency_scores / 1.5)) * detected_cells_mask

    if stats_print:
        # Just print some statistcs for fun
        print("\n\nSignals at squashed intensity level. High to low:")
        print(np.sum(squashed_intensity >= 0.9))
        print(np.sum(squashed_intensity >= 0.8) - np.sum(squashed_intensity >= 0.9))
        print(np.sum(squashed_intensity >= 0.7) - np.sum(squashed_intensity >= 0.8))
        print(np.sum(squashed_intensity >= 0.6) - np.sum(squashed_intensity >= 0.7))
        print(np.sum(squashed_intensity >= 0.5) - np.sum(squashed_intensity >= 0.6))
        print(np.sum(squashed_intensity >= 0.4) - np.sum(squashed_intensity >= 0.5))
        print(np.sum(squashed_intensity >= 0.3) - np.sum(squashed_intensity >= 0.4))
        print(np.sum(squashed_intensity >= 0.2) - np.sum(squashed_intensity >= 0.3))
        print(np.sum(squashed_intensity >= 0.1) - np.sum(squashed_intensity >= 0.2))
        print(np.sum(squashed_intensity > 0.0) - np.sum(squashed_intensity >= 0.1))
        print("\n\nSignals at squashed persistency level. High to low:")
        print(np.sum(squashed_persistency >= 0.9))
        print(np.sum(squashed_persistency >= 0.8) - np.sum(squashed_persistency >= 0.9))
        print(np.sum(squashed_persistency >= 0.7) - np.sum(squashed_persistency >= 0.8))
        print(np.sum(squashed_persistency >= 0.6) - np.sum(squashed_persistency >= 0.7))
        print(np.sum(squashed_persistency >= 0.5) - np.sum(squashed_persistency >= 0.6))
        print(np.sum(squashed_persistency >= 0.4) - np.sum(squashed_persistency >= 0.5))
        print(np.sum(squashed_persistency >= 0.3) - np.sum(squashed_persistency >= 0.4))
        print(np.sum(squashed_persistency >= 0.2) - np.sum(squashed_persistency >= 0.3))
        print(np.sum(squashed_persistency >= 0.1) - np.sum(squashed_persistency >= 0.2))
        print(np.sum(squashed_persistency > 0.0) - np.sum(squashed_persistency >= 0.1))
        print("\n\nTotal signals detected:")
        print(np.sum(detected_cells_mask > 0.0))
        print("\n\nAverage signals per droplet:")
        print(np.sum(detected_cells_mask > 0.0) / len(detected_droplets))
        print('')
        print('')
   
    return detected_cells_mask, intensity_scores, persistency_scores