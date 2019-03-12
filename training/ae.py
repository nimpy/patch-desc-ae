from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
import keras

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os import system
import os
import random

import imageio

img_width, img_height = 16, 16

nb_epoch = 50
batch_size = 32

base_dir = '/home/niaki/Code/ImageNet/tiny-imagenet-200'

train_data_dir = base_dir + '/tiny_train16'
validation_data_dir = base_dir + '/tiny_validation16'
test_data_dir = base_dir + '/tiny_test16'

def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)


def load_data():

    nb_train_samples = 0
    train_folders = os.listdir(train_data_dir)
    train_folders.sort()
    for folder in train_folders:
        nb_train_samples += len(os.listdir(train_data_dir + '/' + folder))
    print(nb_train_samples, "training patches")

    nb_validation_samples = 0
    validation_folders = os.listdir(validation_data_dir)
    validation_folders.sort()
    for folder in validation_folders:
        nb_validation_samples += len(os.listdir(validation_data_dir + '/' + folder))
    print(nb_validation_samples, "validation patches")

    return nb_train_samples, nb_validation_samples


def create_model_128():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    input_img = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)
    # x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    # encoded = MaxPooling2D((2, 2), padding='same')(x)

    # x = Conv2D(16, (3, 3), activation="relu", padding="same")(encoded)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.summary()

    return autoencoder


def create_model_32():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    input_img = Input(shape=input_shape)

    x = Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder


def create_image_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255  # ,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)
    return train_generator, validation_generator


def train_autoencoder(model_version):

    nb_train_samples, nb_validation_samples = load_data()
    train_generator, validation_generator = create_image_generators()

    autoencoder = create_model_128()

    # or, alternatively:
    # autoencoder = load_model(base_dir + '/autoencoder' + model_version_pretrained + '.h5')


    os.system('mkdir ' + base_dir + '/weights' + model_version)

    checkpointer = keras.callbacks.ModelCheckpoint(
        base_dir + '/weights' + model_version + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
        verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    autoencoder.fit_generator(
            fixed_generator(train_generator),
            steps_per_epoch=nb_train_samples,
            epochs=nb_epoch,
            validation_data=fixed_generator(validation_generator),
            validation_steps=nb_validation_samples,
            callbacks=[checkpointer]
            )
    autoencoder.save(base_dir + '/autoencoder' + model_version + '.h5')

    # autoencoder = load_model(base_dir + '/autoencoder' + model_version + '.h5')

def main():
    train_autoencoder('proba_temp0')

if __name__ == "__main__":
    main()