import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Img
import os
import gc
import glob
from tqdm.notebook import tqdm

def load_model(path = "/content/drive/My Drive/LDEO/encoder_weight.pth"):
    """
    a wrapper for load encoder weights
    arg:
        path: the path of encoder_weight.pth
    return:
        model: torch.nn.module, the encoder itself
    """
    assert path.split("/")[-1] == "encoder_weight.pth"
    _encoder = encoder()
    _encoder.load_state_dict(torch.load(path), strict=False)
    _encoder = _encoder.double()
    return _encoder

class encoder(torch.nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size = 3)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size = 3)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size = 3)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size = 3)
        self.mp = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = self.relu(self.mp(self.conv3(out)))
        out = self.relu(self.mp(self.conv4(out)))
        return out

@torch.no_grad()
def load_images_and_encode_one_folder(encoder,
                                      preprocessing = transforms.ToTensor(),
                                      path = "",
                                      device = "cpu"):
    """
    load all the images under specified path,
    during loading, all the images will be resized to 64*64, switch to tensor,
    then only the encoded form will be returned considering the memory issue

    arg:
        encoder: torch.nn.module instance, the encoder of the images trained
        preprocessing: torchvision.transforms instance
        path: str, the path to a specific folder
    """
    # if possible, run it on cuda
    encoder = encoder.to(device)
    # get all the pngs under this folder
    _paths = glob.glob(os.path.join(path, "*.png"))
    # read in the images, preprocessing it
    _img_list = torch.stack([preprocessing(Img.open(img).resize((64,64))) for img in _paths])
    # for efficiency consideration, only save the encoded images and their path
    _encoded = encoder(_img_list.double().to(device))
    # return as normal np.ndarray
    _np_encoded = _encoded.view(_encoded.shape[0],-1).detach().cpu().numpy()
    return _paths, _np_encoded

@torch.no_grad()
def load_encoded_images(encoder,
                        preprocessing = transforms.ToTensor(),
                        path_list = [],
                        device = "cpu"):
    """
    load all the images under specified path,
    during loading, all the images will be resized to 64*64, switch to tensor,
    then only the encoded form will be returned considering the memory issue

    arg:
        encoder: torch.nn.module instance, the encoder of the images trained
        preprocessing: torchvision.transforms instance
        path_list: list, paths of extracted_images folders
    """
    _img_list = None
    _encoded_img_list = None
    _image_path = []
    # if possible, run it on cuda
    encoder = encoder.to(device)
    # for all the extracted_image folder
    for _extracted_image in tqdm(path_list, desc = "loading the image", leave = True):
        # for all the .tif subfolder
        for tif_name in tqdm(glob.iglob(os.path.join(_extracted_image, "*.tif")),
                             desc = _extracted_image.split("/")[-2],
                             leave = False):
            # get all the pngs under this tif folder
            _paths = glob.glob(os.path.join(tif_name, "*.png"))
            # read in the images, preprocessing it
            _img_list = torch.stack([preprocessing(Img.open(img).resize((64,64))) for img in _paths])
            # for efficiency consideration, only save the encoded images and their path
            _encoded = encoder(_img_list.double().to(device))
            # return as normal np.ndarray
            _np_encoded = _encoded.view(_encoded.shape[0],-1).detach().cpu().numpy()
            # add the new paths
            _image_path += _paths
            if not isinstance(_encoded_img_list, np.ndarray):
                _encoded_img_list = _np_encoded
            else:
                _encoded_img_list = np.concatenate((_encoded_img_list, _np_encoded))
            gc.collect()
    return _image_path, _encoded_img_list

def load_process(image_path_list,
                 encoder_path = "/content/drive/My Drive/LDEO/encoder_weight.pth",
                 preprocessing = transforms.ToTensor()):
    """
    wrapper for the whole loading process
    Args:
    image_path_list: list of str, for memory consideration, the image path will be a list of images generate by
    e.g. _extracted_image_list = glob.glob("/content/KORUS/**/extracted_images", recursive = True)
    for users can easily control the length they read in

    encoder_path: str: the path of the encoder
    preprocessing: torchvision.transforms instance, the preprocessing to be applied
    """

    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _encoder = load_model(path = encoder_path)
    _image_path, _encoded_img_list = load_encoded_images(encoder = _encoder,
                                                         preprocessing = preprocessing,
                                                         path_list = image_path_list,
                                                         device = _device)
    print(f"{len(_image_path)} images loaded")
    return _image_path, _encoded_img_list

def load_from_single_folder(path,
                            encoder_path = "/content/drive/My Drive/LDEO/encoder_weight.pth",
                            preprocessing = transforms.ToTensor()):
    """
    wrapper for the whole loading process
    Args:
    path:str, path of a folder filled with .png individual vignettes
    encoder_path: str: the path of the encoder
    preprocessing: torchvision.transforms instance, the preprocessing to be applied
    """
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _encoder = load_model(path = encoder_path)
    _image_path, _encoded_img_list = load_images_and_encode_one_folder(encoder = _encoder,
                                                          path = path,
                                                          device = _device,
                                                          preprocessing = preprocessing)
    return _image_path, _encoded_img_list
