import numpy as np
from keras.models import load_model
from glob import iglob
import os

def load_models(path):
    """
    load all the model under specific path
    path: the path of the folder saving models
    """
    return [load_model(model_path) for model_path in iglob(os.path.join(path, "*.hdf5"))]


def model_predict(obj, models, labelencoder, mode = "average"):
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
        final_pred = labelencoder.inverse_transform(np.argmax(np.mean(predictions, axis = 0), axis = 1))
        pred_prob = np.max(np.mean(predictions, axis = 0), axis = 1)
    else:
        final_pred = labelencoder.inverse_transform(np.argmax(np.max(predictions, axis = 0), axis = 1))
        pred_prob = np.max(np.max(predictions, axis = 0), axis = 1)

    return final_pred, pred_prob
