import csv
import cv2
import pandas as pd 
import matplotlib
import numpy as np 
from matplotlib import image as mpimg
import tensorflow as tf
import numpy as np 
from sklearn.utils import shuffle
from PIL import Image
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers.core import Lambda, Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU


def preprocess(img):
    new_img = img[50:140,:,:]
    new_img = cv2.GaussianBlur(new_img,(5,5),0)
    new_img = cv2.resize(new_img, (200,66), interpolation= cv2.INTER_AREA)
    return new_img

def read_data(batch_size=256):
    """
    Generator function to load driving logs and input images.
    """
    while 1:
        with open('data/driving_log.csv') as driving_log_file:
            driving_log_reader = csv.DictReader(driving_log_file)
            count = 0
            inputs = []
            targets = []
            try:
                for row in driving_log_reader:
                    steering_offset = 0.3

                    centerImage = preprocess(mpimg.imread('data/'+ row['center'].strip()))
                    flippedCenterImage = np.fliplr(centerImage)
                    centerSteering = float(row['steering'])

                    leftImage = preprocess(mpimg.imread('data/'+ row['left'].strip()))
                    flippedLeftImage = np.fliplr(leftImage)
                    leftSteering = centerSteering + steering_offset

                    rightImage = preprocess(mpimg.imread('data/'+ row['right'].strip()))
                    flippedRightImage = np.fliplr(rightImage)
                    rightSteering = centerSteering - steering_offset

                    if count == 0:
                        inputs = np.empty([0, 66, 200, 3], dtype=float)
                        targets = np.empty([0, ], dtype=float)
                    if count < batch_size:
                        inputs = np.append(inputs, np.array([centerImage, flippedCenterImage, leftImage, flippedLeftImage, rightImage, flippedRightImage]), axis=0)
                        targets = np.append(targets, np.array([centerSteering, -centerSteering, leftSteering, -leftSteering, rightSteering, -rightSteering]), axis=0)
                        count += 6
                    else:
                        yield inputs, targets
                        count = 0
            except StopIteration:
                pass

batch_size = 256
use_transfer_learning = False

# define model
if use_transfer_learning:
    model = load_model('model.h5')
else:
	model = Sequential()
	# Layer 1 normalize
	model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
	# Layer 2 convolutional layer 
	model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid'))
	model.add(ELU())
	# Layer 3 convolution layer
	model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid'))
	model.add(ELU())
	# Layer 4 convolutional layer
	model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid'))
	model.add(ELU())
	# Layer 5 convolution layer
	model.add(Convolution2D(64,3,3,  border_mode='valid'))
	model.add(ELU())
	# Layer 6 convolution layer
	model.add(Convolution2D(64,3,3, border_mode='valid'))
	model.add(ELU())
	# Layer 7 Flatten
	model.add(Flatten())
	# Layer 8 dropout
	model.add(Dropout(0.5))
	# Layer 9 fully connected layer
	model.add(Dense(100))
	model.add(ELU())
	# Layer 10 
	model.add(Dense(50))
	model.add(ELU())
	# Layer 11
	model.add(Dense(10))
	model.add(ELU())
	# Layer 12
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	model.summary()

	model.fit_generator(read_data(), samples_per_epoch=8000*6, nb_epoch=1)
	model.save('model.h5')