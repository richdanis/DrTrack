import cv2 as cv
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
from . import manual_circle_hough, cell_detector, droplet_retriever
from pathlib import Path


class Droplet():
    def __init__(self, droplet_id, frame, center_y, center_x, radius, nr_cells, image_name,DROPLET_PATH) -> None:
        self.id = int(droplet_id)
        self.frame = int(frame)
        self.center_y = int(center_y)
        self.center_x = int(center_x)
        self.radius = int(radius)
        self.nr_cells = int(nr_cells)
        self.cells = []
        self.image_name = image_name
        self.folder_path = DROPLET_PATH

    def get_droplet_id(self):
        return self.id
    
    def get_frame(self):
        return self.frame
    
    def get_center(self):
        return (self.center_y, self.center_x)
    
    def get_radius(self):
        return self.radius
    
    def get_nr_cells(self):
        return self.nr_cells
    
    def has_cell(self) -> bool:
        return self.nr_cells > 0
    
    def set_patch(self):
        patch = np.load(Path(self.folder_path / self.image_name / ("f" + str(self.frame) + "_d" + str(self.id).zfill(4) + '.npy')))
        patch = np.float64(droplet_retriever.resize_patch(patch, 40) * (2.0 ** (-16)))
        self.patch = patch

    def get_patch(self):
        return self.patch

    def set_max_brightness(self):
        self.max_brightness = np.max(self.patch)

    def get_max_brightness(self):
        return self.max_brightness
    
    def get_patch_minus_mean(self):
        return self.patch - np.mean(self.patch, axis=0)
    
    def get_patch_size(self):
        return self.patch.shape
    
    def add_cell(self,cell):
        self.cells.append(cell)

    def get_cells(self):
        return self.cells
    
    def get_integrated_brightness_of_cells(self):
        return np.sum([cell.get_integrated_brightness() for cell in self.cells])
    
    def set_hog(self):
        winSize = (64, 64)
        blockSize = (64, 64)
        blockStride = (64, 64)
        cellSize = (32, 32)
        patch = np.uint8(droplet_retriever.resize_patch(self.patch, 64) // 256)
        hog = cv.HOGDescriptor(_winSize=winSize, _blockSize=blockSize, _blockStride=blockStride, _cellSize=cellSize, _nbins=9)
        self.hog_bf = hog.compute(patch[0, :, :])
        self.hog_dapi = hog.compute(patch[1, :, :])

    def get_hog_bf(self):
        return self.hog_bf

    def get_hog_dapi(self):
        return self.hog_dapi

    def set_embedding(self):
        assert False, "Not implemented - correct path to embedding"
        embedding = np.load(Path(self.folder_path / ("f" + str(self.frame) + "_d" + str(self.id).zfill(4))))
        self.embedding = embedding

    def get_embedding(self):
        return self.embedding
    
    def get_pandas_series_without_cell_information(self):
        return pd.Series({"droplet_id": self.droplet_id,
                          "frame": self.frame,
                          "center_y": self.center_y,
                          "center_x": self.center_x,
                          "radius": self.radius,
                          "nr_cells": self.nr_cells})
    
    def safe_as_jason(self):
        return {"droplet_id": self.droplet_id,
                "frame": self.frame,
                "center_y": self.center_y,
                "center_x": self.center_x,
                "radius": self.radius,
                "nr_cells": self.nr_cells,
                "cells": [cell.safe_as_jason() for cell in self.cells]}
        

class Cell():
    def __init__(self, center_y, center_x, intensity_score, persistence_score) -> None:
        self.center_y = center_y
        self.center_x = center_x
        self.intensity_score = intensity_score
        self.persistence_score = persistence_score

    def get_cell_center(self):
        return (self.center_y, self.center_x)
    
    def get_intensity_score(self):
        return self.intensity_score
    
    def get_persistence_score(self):
        return self.persistence_score
    
    def get_integrated_brightness(self):
        return self.intensity_score * self.persistence_score
    
    def relative_position_to_droplet(self, droplet):
        return np.asarray([self.center_y - droplet.get_center()[0], self.center_x - droplet.get_center()[1]])
    
    def resize_patch(self, size):
        pass

    def get_pandas_series(self):
        return pd.Series({"center_y": self.center_y,
                          "center_x": self.center_x,
                          "intensity_score": self.intensity_score,
                          "persistence_score": self.persistence_score})
    
    def safe_as_jason(self):
        return {"center_y": self.center_y,
                "center_x": self.center_x,
                "intensity_score": self.intensity_score,
                "persistence_score": self.persistence_score}


def get_droplet_output(bf_image, refine, radius_min = 12, radius_max = 25):
    droplet_mask, droplet_circles = manual_circle_hough.manual_circle_hough(bf_image, refine, radius_min = radius_min, radius_max = radius_max)


# The input image should be an ndarray with shape (f,c,h,w) where f = frames, c = channels, h = height and w = width of the image.
# IMPORTANT: Datatype should be uint16 just as with the raw images and BF and DAPI must be channels Nr 0 and 1 respectively
def generate_output_from_ndarray(cfg, input_image, output_string_droplets, output_string_cells, refine, optional_output_directory, optional_output, radius_min = 12, radius_max = 25):
    nr_frames = input_image.shape[0]
    nr_channels = input_image.shape[1]

    disable_tqdm = True
    if cfg.verbose == True:
            disable_tqdm = False

    droplets = []
    cells_dict = []
    for frame_nr in tqdm(range(nr_frames), disable=disable_tqdm):

        dapi_channel = input_image[frame_nr, 1, :, :]
        bf_channel = input_image[frame_nr, 0, :, :]
        visualization_channel = np.zeros(bf_channel.shape, dtype = np.float32)

        circles_in_frame = manual_circle_hough.manual_circle_hough(bf_channel, refine, bf_is_inverted = True, radius_min = radius_min, radius_max = radius_max)

        cells_mask, cells_intensities, cells_persistencies = cell_detector.cell_detector(cfg, dapi_channel, bf_channel, circles_in_frame)

        intensities_vector = cells_intensities[cells_mask == 1.0]
        persistence_vector = cells_persistencies[cells_mask == 1.0]

        intens_thresh = np.quantile(intensities_vector, 0.2)
        
        presis_thresh = np.quantile(persistence_vector, 0.2)

        visualization_channel = cv.morphologyEx(cells_mask, cv.MORPH_DILATE, np.ones((3,3)))

        cell_id_counter = 0

        disable = True
        if cfg.verbose == True:
            disable = False

        for id, circ in tqdm(enumerate(circles_in_frame), disable=disable_tqdm):
            center = np.asarray([circ[0], circ[1]])
            radius = circ[2]
            patch_x = (max(int(center[0]) - radius - 2, 0), min(int(center[0]) + radius + 2, cells_mask.shape[0] - 1))
            patch_y = (max(int(center[1]) - radius - 2, 0), min(int(center[1]) + radius + 2, cells_mask.shape[1] - 1))
            local_cells_mask = cells_mask[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            local_cells_intens = cells_intensities[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            local_cells_pers = cells_persistencies[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]

            local_mask = np.zeros(local_cells_mask.shape)
            center_in_patch = center - np.asarray([max(int(center[0]) - radius - 2, 0), max(int(center[1]) - radius - 2, 0)])
            cv.circle(local_mask, np.flip(center_in_patch), radius, 1.0 , -1)
            local_cells_mask = local_cells_mask * local_mask
            local_cells_intens = local_cells_intens * local_mask
            local_cells_pers = local_cells_pers * local_mask

            nr_cells_estimated = np.sum(np.logical_and((local_cells_pers > presis_thresh), (local_cells_intens > intens_thresh)))
            cv.circle(visualization_channel, np.flip(center), radius, 1.0, 1)
            droplets.append({"droplet_id": id, "frame": frame_nr, "center_y": circ[0], "center_x": circ[1], "radius": circ[2], "nr_cells": nr_cells_estimated})
            cell_coords = np.transpose(np.asarray(np.where(local_cells_mask != 0.0)))
            for coord in cell_coords:
                global_center = coord + np.asarray([max(int(center[0]) - radius - 2, 0), max(int(center[1]) - radius - 2, 0)])
                cells_dict.append({"cell_id": cell_id_counter,
                                   "droplet_id": id,
                                   "frame": frame_nr,
                                   "center_y": global_center[0],
                                   "center_x": global_center[1],
                                   "intensity_score": local_cells_intens[coord[0], coord[1]],
                                   "persistence_score": local_cells_pers[coord[0], coord[1]]})
                cell_id_counter = cell_id_counter + 1
        if optional_output:
            to_display = np.float32(np.transpose(np.asarray([visualization_channel * 1, (bf_channel - bf_channel.min()) / (bf_channel.max() - bf_channel.min()), 1.0 * (dapi_channel - dapi_channel.min()) / (dapi_channel.max() - dapi_channel.min())]), [1, 2, 0]))
            cv.imwrite(optional_output_directory + 'detection_visualization_frame_' + str(frame_nr) + '.tiff', to_display)
    
    droplet_df = pd.DataFrame(droplets)
    droplet_df.to_csv(output_string_droplets, index = False)

    # cell_df = pd.DataFrame(cells_dict)
    # cell_df.to_csv(output_string_cells, index = False)


def main():

    parser = argparse.ArgumentParser(description='Droplets and Cells Processing')

    parser.add_argument('--imgname', required=True, help='Image name')
    parser.add_argument('--imgdir', required=True, help='Image directory')
    parser.add_argument('--outdir', help='Output directory (optional, defaults to input directory)')
    parser.add_argument('--outid', help='Output ID (optional, will append to output table names)')
    parser.add_argument('-r','--refine', action='store_true', help='Use refined droplet detection')
    parser.add_argument('-o','--optional', action='store_true', help='Generate output images')

    args = parser.parse_args()

    imgname = args.imgname
    output_path = args.outdir
    output_id = '_id' + args.outid if args.outid else ''

    if not output_path:
        output_path = args.imgdir
        print('Output Directory set to: ' + output_path)

    complete_image_path = args.imgdir + imgname + '.nd2'
    complete_output_path_droplets = output_path + imgname + '_droplets' + output_id + '.csv'
    complete_output_path_cells = output_path + imgname + '_cells' + output_id + '.csv'

    print('Input file is ' + complete_image_path)
    print('Output file for droplets is ' + complete_output_path_droplets)
    print('Output file for cells is ' + complete_output_path_cells)

    # Call the generate_output function with the necessary arguments
    # generate_output(complete_image_path, complete_output_path_droplets, complete_output_path_cells, args.refine, output_path, args.optional)

if __name__ == "__main__":
    main()
