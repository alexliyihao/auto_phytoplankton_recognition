import PIL
import PIL.image as Img
import numpy as np

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

def preprocess(image, to_size):
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
