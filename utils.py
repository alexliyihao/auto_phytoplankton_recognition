import numpy as np
import glob
import os
from preprocess import preprocess

def find_valid_tif(root_path):
    """
    return the list of a valid tif under the root path

    arg:
        root_path: string, the root path start searching
    return:
        list of string, the path of all valid string under this path
    """
    return [i for i in glob.iglob(os.path.join(root_path, "/**/*.tif"), recursive = True) if "bin" not in i and "cal_image" not in i]



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
