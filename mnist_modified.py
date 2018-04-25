

# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd


from PIL import Image
from keras.preprocessing.image import img_to_array

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

K.set_image_dim_ordering('th')


#################################################
###    FROM PROJECT
#################################################

import scipy.io as sio
def read_wiki_images():
    #read wiki metadata file
    meta_info = sio.loadmat('wiki.mat')
    image_content = meta_info['wiki'].item(0)
    required_data = np.column_stack((image_content[2][0],image_content[3][0]))
    label = label = required_data[:,1]
    image_path_list = required_data[:,0]
    images = np.zeros([0,1,28,28])
    image_size_28 = (28,28)
    for item in image_path_list:
        image_file_name = item[0]
        image_file_path = os.path.join("/home/anisha/test/tcss555/training/", "image", image_file_name)
        im = Image.open(image_file_path)
        im=im.convert('L') #makes it greyscale
        im = im.resize(image_size_28, Image.ANTIALIAS)		#opening the image and resizing it to 28*28
        image_to_array = img_to_array(im)
        '''
        print("Image properties:")
        print(type(image_to_array))
        print(image_to_array.shape)
        print(image_to_array.view())
        
        modified_image = im.getdata()
        print(type(modified_image))
        print(modified_image)
        '''
        images = np.concatenate((images, image_to_array[np.newaxis]))
    #print("Images list:")
    images = images.round(decimals=0)
    #print(images.shape)
    return (images, label)

def partition_data_into_test_and_train(input, number_of_test_ids):
    # arranging data in indices to split
    all_Ids = np.arange(len(input.userid))

    # test and train ids
    random.shuffle(all_Ids)
    test_Ids = all_Ids[0:number_of_test_ids]
    train_Ids = all_Ids[number_of_test_ids:]

    # test and train data
    data_test = input.loc[test_Ids, :]
    data_train = input.loc[train_Ids, :]
    
    return (data_test, data_train)

#########################################################################


def getImageFromInput(input):
    #label = input.as_matrix(columns=["gender"])
    label = np.asarray(input["gender"])
    images = np.zeros([0,1,28,28])
    image_size_28 = (28,28)
    #TODO loop on the basis of file name and not userid
    for id in input.userid:
        image_file_name = id + ".jpg"
        image_file_path = os.path.join("/home/anisha/test/tcss555/training/", "image", image_file_name)
        im = Image.open(image_file_path)
        im=im.convert('L') #makes it greyscale
        im = im.resize(image_size_28, Image.ANTIALIAS)		#opening the image and resizing it to 28*28
        image_to_array = img_to_array(im)
        '''
        print("Image properties:")
        print(type(image_to_array))
        print(image_to_array.shape)
        print(image_to_array.view())
        
        modified_image = im.getdata()
        print(type(modified_image))
        print(modified_image)
        '''
        images = np.concatenate((images, image_to_array[np.newaxis]))
    #print("Images list:")
    images = images.round(decimals=0)
    #print(images.shape)
    return (images, label)


print("Reading input")

#read matlab file


#read input and update data frame
profile_file_path = os.path.join("/home/anisha/test/tcss555/training/", "profile/profile.csv")
input = pd.read_csv(profile_file_path, usecols=['userid', 'gender'])

#we partition first and then create image numpy array
print("Partitioning input")
(image_test_dataframe, image_train_dataframe) = partition_data_into_test_and_train(input, 3)

(Images_train, gender_train) = getImageFromInput(image_train_dataframe)
(Images_test, gender_test) = getImageFromInput(image_test_dataframe)


###########################################################################

###########################################################################
###    PROJECT CODE ENDS
###########################################################################

print("")
print("Images_train properties:")
print(Images_train.shape)
print("gender_train properties:")
print(gender_train.shape)
#print(Project_X_train.view())

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
# TODO here 1 is the grey scale channel. We need 3 here for RGB
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

print("")
print("X_train properties:")
print(X_train.shape)
print("y_train properties:")
print(y_train.shape)
print(y_train.view())

# normalize inputs from 0-255 to 0-1
Images_train = Images_train / 255
Images_test = Images_test / 255

# one hot encode outputs
gender_train = np_utils.to_categorical(gender_train)
gender_test = np_utils.to_categorical(gender_test)
num_classes = gender_test.shape[1]



# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.shape[1]


def baseline_model():
	# create model
	# TODO change the input shape for RGB
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# build the model
model = baseline_model()
# Fit the model
model.fit(Images_train, gender_train, validation_data=(Images_test, gender_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(Images_test, gender_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

