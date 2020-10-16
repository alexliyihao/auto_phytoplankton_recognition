import numpy as np
import glob
import os
import numpy as np
from PIL import Image
from os import mkdir, walk, listdir, getcwd
from os.path import join, exists


def extract_image(image):
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

    assert type(tif) in [str, np.ndarray, PIL.TiffImagePlugin.TiffImageFile]
    # if given a path, the input should be opened as a PIL
    if type(tif) == str:
      tif = np.array(Img.open(tif))
    elif type(tif) == PIL.TiffImagePlugin.TiffImageFile:
      tif = np.array(tif)
    #otherwise it is a np.ndarray
    #extract the np.ndarray
    extracted = extract_image(tif)
    return extracted

def model_predict(obj, models, mode = "average"):
    """
    pass object to classification model/model group and generate the result
    Arg:
      obj: np.ndarray/iterable of np.ndarray, the object/objects to be passed to models
      models: model/iterable of models
      mode: str = "average" or "max", the method for ensembling
    return:
      final_pred: int, the prediction
      pred_prob: float, the softmax probability
    """
    assert mode in ['average', "max"]
    # convert the obk into a n*c*h*w np.ndarray
    obj = np.array(obj)
    if len(obj.shape) == 3:
      obj = np.expand_dims(image, 0)

    # if models is a iterable one
    try:
      iter(models)
    # if not, make it in a list
    except TypeError:
      models = [models]
    # pass all the objects to all the models, get a prediction
    predictions = np.stack([model.predict(obj) for model in models], axis = 0)

    # obtain the prediction and probability
    if mode == "average":
        final_pred = np.argmax(np.mean(predictions, axis = 0), axis = 1)
        pred_prob = np.max(np.mean(predictions, axis = 0), axis = 1)
    else:
        final_pred = np.argmax(np.max(predictions, axis = 0), axis = 1)
        pred_prob = np.max(np.max(predictions, axis = 0), axis = 1)

    return final_pred, pred_prob
