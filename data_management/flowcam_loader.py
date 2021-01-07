import glob2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import cv2
import PIL.Image as Img

class flowcam_loader():
    """
    the basic loading unit working with an individual flowcam output folder
    """
    def __init__(self, path = None):
        """
        init of this loader, please be noticed that the initialization of this
        loader will not load any data into memory for efficiency consideration

        Args:
          folder_path: str, the path of the folder
        """
        self._folder_path = path
        self._feature_data = None
        self._tif_paths = self._tif_management(path = path)
        self._csv_path = self._csv_management(path = path)
        self._images = None
        self._features = None

    def _extract_index(self, path):
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
        _tifs = {self._extract_index(i):{"original":i} for i in _tif_paths if "_bin.tif" not in i}
        _binary_mask = [i for i in _tif_paths if "_bin.tif" in i]
        for i in _binary_mask:
            _tifs[self._extract_index(i)]["binary"]=i
        return list(_tifs.values())

    def _csv_management(self, path):
        """
        given a folder path, get the path of the CSV file recording features
        Args:
            path: str, the path string
        Returns:
            str, the data_export.csv's path in the folder
        """
        return [i for i in glob2.iglob(os.path.join(path, "data_export.csv"))][0]

    def _extract_images(self, image):
        """
        extract all the images into the loader
        the image will be extracted from the normal reading order
        Args:
            image: np.ndarray, the tif collage
        Returns:
            _extracted_list: list[np.ndarray], the list of images extracted from the list
        """
        _shape_height = image.shape[0]#the height of the image
        _shape_width = image.shape[1]#the width of the image

        #search for the first column that is not black
        _x = np.min(np.nonzero(np.any(image, axis = (1,2))))

        # search along this column, we can obtain the start pixel row of each image row
        _row_mask = np.any(image[:,_x,:], axis = 1)
        _row_start_y = np.insert(np.nonzero(np.diff(_row_mask.astype(int)) == 1)[0]+1, 0, 0)
        # to be used in later loop
        _row_start_y = np.append(_row_start_y, _shape_height-1)
        _extracted_list = []
        #Search by all the starting rows
        for _y_index in range(_row_start_y.size-1):
            _starting_y = _row_start_y[_y_index]

            # we can find all the start and end column of each image
            # (i.e. they are horizonally bounded)
            _column_mask = np.any(image[_starting_y], axis = 1) # all the non-black point
            _column_start_x = np.insert(np.nonzero(np.diff(_column_mask.astype(int)) == 1)[0]+1, 0, 0)
            _column_end_x = np.nonzero(np.diff(_column_mask.astype(int)) == -1)[0]+1
            # check each horizontally bounded region
            for _x_index in range(_column_start_x.size):
                # check the leftmost column
                _image_starting_x = _column_start_x[_x_index]

                #search in this column between two starting y
                _image_y_mask = np.any(image[_starting_y:_row_start_y[_y_index + 1],_image_starting_x], axis=1)

                #get the end point of y
                _image_end_y = np.where(np.diff(_image_y_mask.astype(int)) == -1)[0][0]+1

                #append the image to the final list
                _extracted_list.append(
                        image[_starting_y:_starting_y+_image_end_y,
                              _image_starting_x:_column_end_x[_x_index]]
                        )
        return _extracted_list

    def load_features(self):
        """
        load the csv files of the folder into the loader
        """
        self._features = pd.read_csv(self._csv_path)

    def load_features_temp(self):
        """
        load the csv files of the folder into the loader but not save into the object
        """
        return pd.read_csv(self._csv_path)

    def load_images(self):
        """
        load the images into the loader,
        all the images are ordered by the official order
        """
        self._images = [self._extract_images(image = np.array(Img.open(i["original"]))) for i in self._tif_paths]

    def load_images_temp(self):
        """
        load the images into the loader but not save into the object
        all the images are ordered by the official order
        """
        return [self._extract_images(image = np.array(Img.open(i["original"]))) for i in self._tif_paths]

    def load(self):
        """
        a wrapper load both images and csv features
        """
        self.load_features()
        self.load_images()

    def load_temp(self):
        """
        a wrapper load both images and csv features but not save into the object
        """
        return self.load_features_temp, self.load_images_temp

    @property
    def images(self):
        return self._images

    @property
    def features(self):
        return self._features

    def __getitem__(self, i):
        """
        a universal wrapper returns individual rows, no matter what is saved loaded
        Args:
          i, the index
        """
        try:
            image = self._images[i]
        except IndexError:
            raise
        except TypeError:
            image = None

        try:
            features = self.features.iloc[i]
        except AttributeError:
            raise
        except TypeError:
            features = None

        return individual_result(image = image, features = features)

class individual_result(dict):
    @property
    def image(self):
        return self["image"]

    @property
    def features(self):
        return self["features"]

    @property
    def plot_image(self):
        plt.imshow(self["image"])
        plt.show()
        return self["image"]
