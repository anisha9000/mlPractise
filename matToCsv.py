import scipy.io
import numpy as np
import pandas as pd
import os
from PIL import Image
from pathlib import Path
from keras.preprocessing.image import img_to_array

image_rgb_file_name = "wiki_images_rgb.npy"
image_l_file_name = "wiki_images_l.npy"
label_rgb_file_name = "wiki_label_rgb.npy"
label_l_file_name = "wiki_label_l.npy"

meta_info = scipy.io.loadmat("wiki.mat")
print("Retrieving new data")
image_content = meta_info['wiki'].item(0)
required_data = np.column_stack((image_content[2][0], image_content[3][0]))
required_data = required_data[~pd.isnull(required_data[:, 1])]
label = required_data[:, 1]
label = 1-label
image_path_list = required_data[:, 0]

image_size_28 = (64, 64)

images_rgb = np.zeros([0, 3, 64,64])
images_l = np.zeros([0, 1, 64,64])
label_rgb = []
label_l = []

counter = 0;
for item in image_path_list:
    print(counter)
    image_file_name = item[0]
    image_file_path = os.path.join("wiki", image_file_name)
    if(Path(image_file_path).is_file()) :
        im = Image.open(image_file_path)
        im = im.resize(image_size_28, Image.ANTIALIAS)  # opening the image and resizing it to 28*28
        image_to_array = img_to_array(im, data_format='channels_first')
        print(image_file_name)
        if(im.mode == 'RGB'):
            images_rgb = np.concatenate((images_rgb, image_to_array[np.newaxis]))
            label_rgb = np.append(label_rgb, label[counter])
        else:
            images_l = np.concatenate((images_l, image_to_array[np.newaxis]))
            label_l = np.append(label_l, [counter])
        counter+=1
images_rgb = images_rgb.round(decimals=0) / 255
images_l = images_l.round(decimals=0) / 255
label_rgb = np.array(label_rgb)
label_l = np.array(label_l)
np.save(image_rgb_file_name, images_rgb)
np.save(image_l_file_name, images_l)
np.save(label_rgb_file_name, label_rgb)
np.save(label_l_file_name, label_l)
