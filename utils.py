import numpy as np
import glob
import os
import PIL
from PIL import Image as Img
from os import mkdir, walk, listdir, getcwd
from os.path import join, exists


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
    
def find_valid_tif(root_path):
    """
    return the list of a valid tif under the root path

    arg:
        root_path: string, the root path start searching
    return:
        list of string, the path of all valid string under this path
    """
    return [i for i in glob.iglob(os.path.join(root_path, "/**/*.tif"), recursive = True) if "bin" not in i and "cal_image" not in i]

def extract_from_tif(tif):
    """
    extract the image from a existed tif, which can be a PIL object, np.ndarray or a path

    arg:
      tif: str, np.ndarray, or PIL.TiffImagePlugin.TiffImageFile, the tif file
    return:
      extracted: list of np.ndarray, the extracted
    """

    # if given a path, the input should be opened as a PIL
    if type(tif) == str:
      tif = np.array(Img.open(tif))
    elif isinstance(tif, np.ndarray) == False:
      tif = np.array(tif)
    #otherwise it is a np.ndarray
    #extract the np.ndarray
    extracted = extract_image(tif)
    return extracted
