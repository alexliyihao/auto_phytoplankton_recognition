import glob2
import os
import numpy as np
import re

class flowcam_loader():
    """
    the basic loading unit working with an individual flowcam output folder
    """
    def __init__(self, path = None):
        """
        init of this loader, please be noticed that the initialization of this
        loader will not load any data into memory.

        Args:
          folder_path: str, the path of the folder
        """
        self._folder_path = path
        self._feature_data =
        self._tif_paths = self._tif_management(path = path)
        self._csv_paths = self._csv_management(path = path)

    def extract_index(self, path):
        """
        given a path_string, extract the batch number in int
        Args:
          path_string: the path of the tif
        Returns:
          int: the number of batch
        """
        return int(re.findall("\d{6}", path)[0])

    def _tif_management(self, path):
        """
        given a folder path, get all the paths of original image and binary masks
        Args:
            path: the path string
        Returns:
            list[dict]: a list of dictionary, sorted by their batch number
                        each element dictionary has path of the binary mask
                        and the original image
        """
        _tif_paths = [i for i in glob2.iglob(os.path.join(path, "*.tif")) if "cal_image" not in i]
        _tifs = {self.extract_index(i):{"original":i} for i in _tif_paths if "_bin.tif" not in i}
        _binary_mask = [i for i in _tif_paths if "_bin.tif" in i]
        for i in _binary_mask:
            _tifs[self.extract_index(i)]["binary"]=i
        return list(_tifs.values())

    def _csv_management(self, path):
        """
        given a folder path, get the path of the CSV file recording features
        """
        return [i for i in glob2.iglob(os.path.join(path, "data_export.csv"))][0]

    def length(self):
        """
        return the current length inside the folder
        """
        return len(self._images)

    def extract_images(self, image):
        """
        extract all the images into the loader, improved from the original one,
        it will use the binary mask to create a white input image.
        """
        shape_height = image.shape[0]#the height of the image
        shape_width = image.shape[1]#the width of the image

        for x in range(shape_width): # scan by column
            if np.any(image[:,x,:]):#if any the pixel is not black
                break #stop scanning

        mask = [np.any(i) for i in image[:,x,:]] # all the non-black point
        row_start_y = [i for i in range(len(mask))\
                      if ((mask[i] == True) and (mask[i+1] == True) and (mask[i-1] == False)) \
                          or ((i == 0) and (mask[i] == True))]

        row_start_y.append(shape_height)

        extracted_list = []

        for y_index in range(len(row_start_y) - 1):#back to rows, check rows
            y = row_start_y[y_index]
            mask = [np.any(i) for i in image[y,:,:]] # all the non-black point
            image_start_x = [i for i in range(len(mask)) \
                            if (((mask[i] == True) and (mask[i-1] == False))\
                                or ((i == 0) and (mask[i] == True)))]
            image_end_x = [i for i in range(1, len(mask)) if ((mask[i] == False) and (mask[i-1] == True))]

            for start_index in range(len(image_start_x)): # back to columns
                image_x = image_start_x[start_index]

                #check the non-black points the column image_x for the row
                image_mask = [np.any(i) for i in image[y:row_start_y[y_index + 1],image_x,:]]

                #get the end point
                image_end_y = [i for i in range(len(image_mask))
                              if ((image_mask[i] == False) and (image_mask [i-1] == True))][0]

                #append the image to the final list
                extracted_list.append(image[y:y+image_end_y,image_x:image_end_x[start_index],:])

        return extracted_list
    def load_features(self):
        """
        load the csv files of the folder into the
        """
        pass
