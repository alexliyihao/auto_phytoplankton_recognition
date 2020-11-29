import torch
from torchvision import transforms
import numpy as np
import os
import glob
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset

def preprocess(img, transformation = transforms.ToTensor(), to_size = (200,200)):
    """
    the function operating preprocessing to individual image
    1. control the size to size of to_size while keep the same aspect ratio
    2. if the image is smaller than to_size, add a black padding
    3. convert it from [0,255] int to [0,1] float format

    Args:
        img: np.array/PIL.Image.Image, individual image
        to_size: tuple(int), the size of output images
        transformatiom: torchvision.transform object
    return :
        preprocessed image in torch.tensor format
    """
    if isinstance(img, np.ndarray):
      img = Image.fromarray(img)
    img.thumbnail(to_size)
    img = np.array(img)
    diff_x = np.ceil((to_size[0] - img.shape[0])/2).astype(int)
    diff_y = np.ceil((to_size[1] - img.shape[1])/2).astype(int)
    padded = np.pad(img/255.0 , pad_width= ((diff_x,diff_x),(diff_y,diff_y),(0,0)))
    return transformation(padded[0:to_size[0], 0:to_size[1]])


def read_classified_data(root_path, to_size = (200,200), transformation = transforms.ToTensor()):
    """
    read in the classified data,
    for we have an additional csv file inside,
    this dataset loader will replace torchvision.datasets.ImageFolder
    arg:
        root_path: the root_path of classified image folder
        to_size: the actual size of all the output
        transformation: the torchvision level preprocessing
    return:
        dataset: torch.utils.data.TensorDataset with:
            images: N*H*W*C tensor
            features: N*num_feature tensor
            labels: (N,1) tensor
        label_dict: dict{int:string} the dictionary of integer label to numerical label
    """
    label_dict = {}
    # for each folder in the dataset
    # get the label
    for i, label in tqdm(enumerate(sorted(os.listdir(root_path))), desc = "Read in...", leave = False):
        if len(os.listdir(sub_path)) == 0:
            continue
        sub_path = os.path.join(root_path, label)
        # write the label in the label dict
        label_dict[i] = label
        # find the csv, there should be one and only one csv
        csv_path = glob.glob(os.path.join(sub_path,"*.csv"))[0]
        df = pd.read_csv(csv_path)
        # the csv should have a image_name list indicating the 1-1 correspondense
        image_origin = df["image_name"]
        # get the rest and the features
        df.drop(labels = "image_name", axis = "columns", inplace = True)
        # concate them to our dataset
        if i == 0:
            features = torch.from_numpy(df.to_numpy())
            images = torch.stack([preprocess(Image.open(os.path.join(sub_path, i)).convert("RGB"),
                                            to_size = to_size,
                                            transformation = transformation) for i in image_origin])
            labels = torch.ones(image_origin.shape[0])*label
        else:
            features = torch.cat((features,torch.from_numpy(df.to_numpy())))
            images = torch.cat(images,torch.stack([preprocess(Image.open(os.path.join(sub_path, i)).convert("RGB"),
                                            to_size = to_size,
                                            transformation = transformation) for i in image_origin]))
            labels = torch.cat(labels, torch.ones(image_origin.shape[0])*label)
    # return the dataset with our label_dict
    return TensorDataset(images,features, labels),label_dict
