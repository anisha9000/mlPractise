import numpy as np
import pandas as pd
import os
from PIL import Image
from keras.preprocessing.image import img_to_array

#read input and update data frame
profile_file_path = os.path.join("/home/anisha/temp/tcss555/training/", "profile/profile.csv")
input = pd.read_csv(profile_file_path)

#get images from directory
images = []
for id in input.userid:
    image_file_name = id + ".jpg"
    image_file_path = os.path.join("/home/anisha/temp/tcss555/training/", "image", image_file_name)
    im = Image.open(image_file_path)
    image_to_array = img_to_array(im)
    print(image_file_name)    
    print(image_to_array)
    images.append(image_to_array)

input["images"] = images



