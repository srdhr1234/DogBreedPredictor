# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:13:54 2023

@author: sridhar.iyer
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception

# import necessary modules
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
# sklearn
from sklearn.datasets import load_files
import glob2

# import necessary modules
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
# sklearn
from sklearn.datasets import load_files
import numpy as np
from keras.utils import np_utils
import numpy as np
import random
from glob import glob
import cv2
from tqdm import tqdm
#from extract_bottleneck_features import *
import os

import os
import urllib.request
import zipfile

# # Download the dataset from the official website
# # url = "https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"
# filename = "dogImages.zip"
# # if not os.path.exists(filename):
#   #  urllib.request.urlretrieve(url, filename)

# # Extract the dataset
# if not os.path.exists("Images"):
#     with zipfile.ZipFile(filename, 'r') as zip_ref:
#         zip_ref.extractall()


# def load_dataset(path):
#     data = load_files(path)
#     dog_files = np.array(data['filenames'])
#     dog_targets = np_utils.to_categorical(np.array(data['target']))#, 133)
#     return dog_files, dog_targets

# # load train, test, and validation datasets
# train_files, train_targets = load_dataset('dogImages/train')
# valid_files, valid_targets = load_dataset('dogImages/valid')
# test_files, test_targets = load_dataset('dogImages/test')

# # load list of dog names
# # the [20:-1] portion simply removes the filepath and folder number
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

print(dog_names)

# # print statistics about the dataset
# print('There are %d total dog categories.' % len(dog_names))
# print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
# print('There are %d training dog images.' % len(train_files))
# print('There are %d validation dog images.' % len(valid_files))
# print('There are %d test dog images.'% len(test_files))

# # define constants
# IMAGE_SIZE = 299
# BATCH_SIZE = 64
# NUM_CLASSES = len(dog_names)

# # define image data generator with rescaling and resizing
# datagen = ImageDataGenerator(rescale=1./255, 
#                              width_shift_range=0.1, 
#                              height_shift_range=0.1, 
#                              rotation_range=20, 
#                              shear_range=0.2,
#                              zoom_range=0.2, 
#                              horizontal_flip=True, 
#                              validation_split=0.2)

# # load the training data
# train_generator = datagen.flow_from_directory('dogImages/train', 
#                                               target_size=(IMAGE_SIZE, IMAGE_SIZE), 
#                                               batch_size=BATCH_SIZE , 
#                                               class_mode='categorical', 
#                                               subset='training')

# # load the validation data
# valid_generator = datagen.flow_from_directory('dogImages/valid', 
#                                               target_size=(IMAGE_SIZE, IMAGE_SIZE), 
#                                               batch_size=BATCH_SIZE , 
#                                               class_mode='categorical', 
#                                               subset='validation')

# # load the test data
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory('dogImages/test',
#                                                   target_size=(IMAGE_SIZE, IMAGE_SIZE),
#                                                   batch_size=BATCH_SIZE ,
#                                                   class_mode='categorical')



# define constants
IMAGE_SIZE = 299
BATCH_SIZE = 64
NUM_CLASSES = len(dog_names)


# define image data generator with rescaling and resizing
datagen = ImageDataGenerator(rescale=1./255, 
                              width_shift_range=0.1, 
                              height_shift_range=0.1, 
                              rotation_range=20, 
                              shear_range=0.2,
                              zoom_range=0.2, 
                              horizontal_flip=True, 
                              validation_split=0.2)

# load the training data
train_generator = datagen.flow_from_directory('dogImages/train', 
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE), 
                                              batch_size=BATCH_SIZE , 
                                              class_mode='categorical', 
                                              subset='training')

# load the validation data
valid_generator = datagen.flow_from_directory('dogImages/valid', 
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE), 
                                              batch_size=BATCH_SIZE , 
                                              class_mode='categorical', 
                                              subset='validation')

# load the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('dogImages/test',
                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                  batch_size=BATCH_SIZE ,
                                                  class_mode='categorical')


# # MODEL COMPILATION
import pickle

# define Xception architecture
xception = Xception(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
xception.trainable = False

model = tf.keras.Sequential([
    xception,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# define callback to save the best model during training
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/weights.best.xception.hdf5', 
                               verbose=1, save_best_only=True)

# define early stopping callback to stop training if there is no improvement in validation accuracy after 10 continuous epochs
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

# # train the model
history = model.fit(train_generator, epochs=5, 
                    validation_data=valid_generator, 
                     callbacks=[checkpointer, earlystopper], 
                    verbose=1)

#with open('history.pkl', 'wb') as f:
 #   pickle.dump(history.history, f)
 
 
pickle.dump(model,open('history.pkl', 'wb'))


# load the best model for testing
model.load_weights('Xception.h5')

# evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print('Test loss: {:.4f}'.format(test_loss))
print('Test accuracy: {:.4f}'.format(test_accuracy))

# function to predict breed of an input image




def predict_breed(img_path, model, class_names):
    # load image using keras.preprocessing.image.load_img
    img = image.load_img(img_path, target_size=(299, 299))
    # preprocess the image using keras.preprocessing.image.img_to_array
    x = image.img_to_array(img)
    # normalize the image pixels by dividing each pixel value by 255
    x = x / 255.0
    # add an extra dimension to the image data to match the input shape of the model
    x = np.expand_dims(x, axis=0)
    # make the prediction using the trained model
    preds = model.predict(x)
    # get the index of the predicted class
    idx = np.argmax(preds)
    # get the predicted class name using the class_names list
    breed_name = class_names[idx]
    # return the predicted breed name
    return breed_name

# # function to plot the training history
# def plot_history(history):
#     # plot training and validation accuracy values
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.show()

#     # plot training and validation loss values
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.show()



# # plot the training history
# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


#Moment of Truth - Predict the Breed

from tensorflow.keras.preprocessing import image

# predict the breed of an input image
img_path = 'dogImages/test/004.Akita/Akita_00244.jpg'
breed_name = predict_breed(img_path, model, dog_names)
print('Predicted breed:', breed_name)


img_path = 'dogImages/test/014.Basenji/Basenji_00955.jpg'
breed_name = predict_breed(img_path, model, dog_names)
print('Predicted breed:', breed_name)

img_path = 'dogImages/test/054.Collie/Collie_03790.jpg'
breed_name = predict_breed(img_path, model, dog_names)
print('Predicted breed:', breed_name)

img_path = 'dogImages/test/124.Poodle/Poodle_07903.jpg'
breed_name = predict_breed(img_path, model, dog_names)
print('Predicted breed:', breed_name)

