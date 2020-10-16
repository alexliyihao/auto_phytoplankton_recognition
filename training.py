import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

def generate_callbacks(tensorboard = False):
    # Callback : Learning Rate annealer
    reduceLR = ReduceLROnPlateau(monitor = 'val_loss',
                                 patience = 15,
                                 factor = 0.5,
                                 min_lr = 0,
                                 verbose = 1)
    # Callback : Save best model
    chkPoint = ModelCheckpoint("model.hdf5",
                              monitor = 'val_loss',
                              save_best_only = True,
                              save_weights_only = False,
                              mode = 'auto',
                              stop_freq = 1,
                              verbose = 0)
    # Callback : Early Stop
    earlyStop = EarlyStopping(monitor='val_loss',
                              mode = 'auto',
                              patience = 400,
                              min_delta = 0.01,
                              verbose = 1)
    if tensorboard == False:
        return [reduceLR,  chkPoint, earlyStop]
    else:
        TensorBoard_callback = TensorBoard(log_dir="./logs")
        return [reduceLR,  chkPoint, earlyStop, TensorBoard_callback]

def create_model(image_size,
                 encoder,
                 model = tf.keras.applications.EfficientNetB2,
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
    input = Input(shape= image_size)
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
