import numpy as np
from . import cell_finding

# takes in the dapi channel, the bf channel and a list of detected droplets
# returns three matrices. The first is a mask that is 1 at a location where a significant peak in the dapi channel has been found
# The second is a matrix that contains some intensity measure of the peaks in the dapi channel.
# The third is a matrix that contains a persistency measure (basically some sort of width of the peaks) of the peaks found in the dapi channel.
def cell_detector(cfg, dapi_channel, bf_channel, detected_droplets):

    cell_detection_scores = cell_finding.cell_finding(cfg, detected_droplets, dapi_channel, bf_channel)
    detected_cells_mask = cell_detection_scores[0, :, :]
    intensity_scores = cell_detection_scores[1, :, :]
    persistency_scores = cell_detection_scores[2, :, :]
   
    return detected_cells_mask, intensity_scores, persistency_scores