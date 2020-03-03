# Machine Learning Homework 4 - Image Classification

__author__ = '**'

# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import sys
import pandas as pd
from collections import Counter
# Keras
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.losses import binary_crossentropy
import seaborn as sns
import os
import sys
import glob
import pickle

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.utils import layer_utils
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, adam, adamax, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ReLU, ELU
from keras.losses import binary_crossentropy
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


### Already implemented
def get_data(datafile):
	dataframe = pd.read_csv(datafile)
	dataframe = shuffle(dataframe)
	data = list(dataframe.values)
	labels, images = [], []
	for line in data:
		labels.append(line[0])
		images.append(line[1:])
	labels = np.array(labels)
	images = np.array(images).astype('float32')
	images /= 255
	return images, labels


### Already implemented
def visualize_weights(trained_model, num_to_display=20, save=True, hot=True):
	layer1 = trained_model.layers[0]
	weights = layer1.get_weights()[0]

	# Feel free to change the color scheme
	colors = 'hot' if hot else 'binary'
	try:
		os.mkdir('weight_visualizations')
	except FileExistsError:
		pass
	for i in range(num_to_display):
		wi = weights[:,i].reshape(28, 28)
		plt.imshow(wi, cmap=colors, interpolation='nearest')
		if save:
			plt.savefig('./weight_visualizations/unit' + str(i) + '_weights.png')
		else:
			plt.show()


### Already implemented
def output_predictions(predictions):
	with open('predictions.txt', 'w+') as f:
		for pred in predictions:
			f.write(str(pred) + '\n')


def plot_history(history):
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    train_acc_history = history.history['acc']
    val_acc_history = history.history['val_acc']

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(x = [i for i in range(len(val_loss_history))], y = train_loss_history, ax = ax1 , mew=1, label = 'train loss')
    sns.lineplot(x = [i for i in range(len(val_loss_history))], y = val_loss_history, ax = ax1 , mew=1, label = 'val loss')
    ax1.legend(loc = 'lower right')
    ax1.set_title('Epoch Loss Plot')


    sns.lineplot(x = [i for i in range(len(val_loss_history))], y = train_acc_history, ax = ax2 , mew=1, label = 'train acc')
    sns.lineplot(x = [i for i in range(len(val_loss_history))], y = val_acc_history, ax = ax2 , mew=1, label = 'val acc')
    ax2.set_title('Epoch Accuracy Plot')
    ax2.legend(loc = 'lower right')

    plt.show()


def create_mlp(args=None):
	# Define model architecture
    model = Sequential()
    model.add(Dense(100, input_shape = (784,), activation = 'relu'))
    model.add(BatchNormalization(momentum = .70))
    model.add(Dropout(0.6))
    model.add(Dense(50, activation = 'relu'))
    model.add(BatchNormalization(momentum = .70))
    model.add(Dense(50, activation = 'relu'))
    model.add(BatchNormalization(momentum = .70))
    model.add(Dense(50, activation = 'relu'))
    model.add(BatchNormalization(momentum = .70))
    model.add(Dense(25, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    model = create_mlp(args)
    with tf.device('/gpu:0'):
        model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=True))

        history = model.fit(x_train,y_train , batch_size= 20,
                       epochs=100, verbose=2, validation_split= .2, callbacks= [early_stopping, MCP])
        sess.close()
        model = load_model(model_path) #Load best model from disc
        return model, history


def create_cnn(args=None):
    # 28x28 images with 1 color channel
    input_shape = (28, 28, 1)

    # Define model architecture
    model = Sequential()
    model.add(
        Conv2D(200, (1, 1),
               activation='relu',
               input_shape=input_shape,
               padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(100, (1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    model.add(Conv2D(50, (1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(512))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    x_train = x_train.reshape(-1, 28, 28, 1)
    # You can use args to pass parameter values to this method
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    model = create_cnn(args)
    with tf.device('/gpu:0'):
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=True))

        history = model.fit(x_train,
                            y_train,
                            batch_size=100,
                            epochs=50,
                            verbose=2,
                            validation_split=.2,
                            callbacks=[early_stopping, MCP])
        sess.close()
        model = load_model(model_path)  #Load best model from disc
    return model, history



if __name__ == '__main__':
	### Before you submit, switch this to grading_mode = False and rerun ###
	grading_mode = True
	if grading_mode:
	# When we grade, we'll provide the file names as command-line arguments
	    if (len(sys.argv) != 3):
	        print("Usage:\n\tpython3 fashion.py train_file test_file")
	        exit()
	    #train_file = 'fashion_train.csv'
	    #test_file = 'fashion_test.csv'
	    train_file, test_file = sys.argv[1], sys.argv[2]
	    x_train, y_train = get_data(train_file)
	    x_test, y_test = get_data(test_file)

	    x_test = x_test.reshape(-1, 28, 28, 1)
	    # train your best model
	    best_model = load_model('best_cnn_ model.h5')
	    # use your best model to generate predictions for the test_file
	    predictions = best_model.predict_classes(x_test)
	    output_predictions(predictions)

		# Include all of the required figures in your report. Don't generate them here.

	else:
		train_file = 'fashion_train.csv'
		test_file = 'fashion_test_labeled.csv'
		# MLP
		mlp_model, mlp_history = train_mlp(x_train, y_train)
		plot_history(mlp_history)
		visualize_weights(mlp_model)

		# CNN
		cnn_model, cnn_history = train_cnn(x_train, y_train)
		plot_history(cnn_history)
