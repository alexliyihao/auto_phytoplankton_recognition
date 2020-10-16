import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from tqdm.keras import TqdmCallback

def generate_callbacks(model_path, tensorboard = False):
    # Callback : Learning Rate annealer
    reduceLR = ReduceLROnPlateau(monitor = 'val_sparse_categorical_accuracy',
                                 patience = 15,
                                 factor = 0.5,
                                 min_lr = 1e-6,
                                 verbose = 1)
    # Callback : Save best model
    if model_path[-5:] != ".hdf5":
        model_path += ".hdf5"
    chkPoint = ModelCheckpoint(model_path,
                               monitor = 'val_sparse_categorical_accuracy',
                               save_best_only = True,
                               save_weights_only = False,
                               save_freq= 1,
                               verbose = 0)

    if tensorboard == False:
        return [reduceLR,  chkPoint, TqdmCallback(verbose=1)]
    else:
        TensorBoard_callback = TensorBoard(log_dir="./logs")
        return [reduceLR,  chkPoint, TensorBoard_callback, TqdmCallback(verbose=1)]

def create_model(image_size,
                 encoder,
                 model,
                 optimizer = tf.keras.optimizers.Adam(lr = 0.05),
                 metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]):
    """
    create a image classification model
    image_size: the size of input image
    encoder: sklearn.preprocessing.LabelEncoder, the encoder of y
    model: tensorflow.python.keras.applications.function the transfer learning model
    optimizer: tensorflow.keras.optimizers, optimizer
    metrics: a list of metrics

    return:
      a compiled tensorflow.python.keras.engine.functional.Functional model
    """
    efn = model(weights='imagenet', include_top = False)
    input = Input(shape= (*image_size,3))
    x = efn(input)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(encoder.classes_.shape[0], activation='softmax')(x)
    model = Model(input,output)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=metrics)
    return model
