import PIL
from PIL import Image as Img
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_classified_dataset(root_path, to_size = (200,200), image_data_generator = None):
    """
    a wrapper to read classified images from a root path
    args:
        root_path: string, the root path of the folders
        to_size: tuple of int, the final size of image
        image_data_generator: tf.keras.preprocessing.image.ImageDataGenerator
    """
    if image_data_generator == None:
        image_data_generator = ImageDataGenerator(rotation_range=360,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     brightness_range=None,
                                     shear_range=0.1,
                                     zoom_range=0.1,
                                     fill_mode="nearest",
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     )
    X,y,labels = generate_dataset(*read_classified_image(root_path = root_path, to_size = to_size),
                                 image_data_generator)
    return tf.data.Dataset.from_tensor_slices((X,y)), labels

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

def preprocess(image, to_size = (200,200)):
    """
    the function operating preprocessing to individual image
    1. control the size to size of to_size while keep the same aspect ratio
    2. if the image is smaller than to_size, add a black padding
    3. convert it from [0,255] int to [0,1] float format

    image: np.array/PIL.Image.Image, individual image
    to_size: tuple

    return :
    a preprocesses
    """
    if type(img) == np.ndarray:
      img = Img.fromarray(img)

    img.thumbnail(to_size)

    img = np.array(img)
    diff_x = np.ceil((to_size[0] - img.shape[0])/2).astype(int)
    diff_y = np.ceil((to_size[1] - img.shape[1])/2).astype(int)
    padded = np.pad(img/255.0 , pad_width= ((diff_x,diff_x),(diff_y,diff_y),(0,0)))
    return padded[0:to_size[0], 0:to_size[1]]

def read_classified_image(root_path, to_size = (200,200)):
    """
    before tfds.folder_dataset.ImageFolder is correctly imported I'll still use mine

    arg:
        root_path: the root_path of classified image folder
        to_size: the actual size of all the output
    return:
        image_list: 2D list of np.ndarray
        label_list: 1D list of string, the labels with 1-1 corrspodense to image_list
    """
    label_list = []
    image_list = []
    for label in tqdm(os.listdir(root_path), desc = "Read in...", leave = False):
      sub_path = os.path.join(path, label)
      if len(os.listdir(sub_path)) == 0:
        continue
      sub_image_list = []
      for j in tqdm(os.listdir(sub_path), desc = f"label {label}", leave = False):
        image = Image.open(os.path.join(sub_path, j))
        image = preprocess(image, to_size = to_size)
        sub_image_list.append(image)
      label_list.append(label)
      image_list.append(sub_image_list)
      return image_list, label_list

def generate_dataset(image_list, label_list, image_data_generator):
    """
    for the image in image_list, if there's any imbalanced dataset,
    use image_data_generator to re-balance it
    """
    majority_size = np.max(np.fromiter((len(sub_list) for sub_list in image_list), dtype = int))
    X = []
    y = []
    for img_list,label in tqdm(zip(image_list, label_list), desc = "balancing the dataset...", leave = False):
        X+=img_list
        for ctr in range(majority_size - len(img_list)):
            X.append(datagen.random_transform(sub_list[np.random.randint(0,len(img_list))]))
        y+=[label]*majority_size

    y = pd.DataFrame(y)
    OHE = OneHotEncoder()
    y = OHE.fit_transform(y).todense()
    return X,y, OHE.categories_


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
