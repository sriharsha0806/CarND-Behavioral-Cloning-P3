from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda 
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam 
from keras.regularizers import l2
import keras.backend as K
from config import *
from load_data import generate_data_batch, split_train_val
import tensorflow as tf 
tf.python.control_flow_ops = tf 

def get_nvidia_model(summary=True):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(NVIDIA_H, NVIDIA_W, CONFIG['input_channels'])))
    model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))
    if summary:
        model.summary()
    return model   

def comma_ai_model(summary=True):
    ch, row, col = 3, 160, 320
    model = Sequential()
    model.add(Lambda(lambda x : x/255.0-0.5, input_shape=(row,col,ch)))
    model.add(Convolution2D(16,8,8, subsample=(4,4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32,5,5, subsample=(2,2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64,5,5, subsample=(2,2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    if summary:
        model.summary()
    return model

if __name__ == '__main__':
    # split udacity csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')
    # get network model and compile it (default Adam opt)
    nvidia_net = get_nvidia_model(summary=True)
    adam = Adam(lr=1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nvidia_net.compile(optimizer=adam, loss='mse')
    nvidia_net.fit_generator(generator=generate_data_batch(train_data, augment_data=True, bias=CONFIG['bias']),
                         samples_per_epoch=300*CONFIG['batchsize'],
                         nb_epoch=5,
                         validation_data=generate_data_batch(val_data, augment_data=False, bias=1.0),
                         nb_val_samples=100*CONFIG['batchsize'])
    nvidia_net.save('model.h5')