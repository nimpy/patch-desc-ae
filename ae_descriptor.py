from keras.layers import Input, MaxPooling2D, Conv2D
from keras.models import Model, load_model

import numpy as np

nr_channels = 3
models_dir = '/home/niaki/Projects/patch-desc-ae/models'


def init_descr(model_version=32, nr_feature_maps_layer1=16, nr_feature_maps_layer23=8, patch_height=16, patch_width=16):

    encoder_trained = load_model(models_dir + '/encoder_' + str(model_version) + '.h5')

    if patch_height == 16 and patch_width == 16:
        return encoder_trained
    else:
        input_shape = (patch_height, patch_width, nr_channels)

        input_img = Input(shape=input_shape)

        x = Conv2D(nr_feature_maps_layer1, (3, 3), activation="relu", padding="same")(input_img)
        x = Conv2D(nr_feature_maps_layer23, (3, 3), activation="relu", padding="same")(x)
        x = Conv2D(nr_feature_maps_layer23, (3, 3), activation="relu", padding="same")(x)
        encoded = MaxPooling2D((8, 8), padding="same")(x)

        encoder = Model(input_img, encoded)

        for i in range(len(encoder.layers)):
            encoder.get_layer(index=i).set_weights(encoder_trained.get_layer(index=i).get_weights())

        return encoder


# the only two parameter options available (i.e. trained)
# -- the two versions of descriptor using 32 and 128 units for the code layer

def init_descr_32(patch_height=16, patch_width=16):
    return init_descr(model_version=32, nr_feature_maps_layer1=16, nr_feature_maps_layer23=8, patch_height=patch_height, patch_width=patch_width)


def init_descr_128(patch_height=16, patch_width=16):
    return init_descr(model_version=128, nr_feature_maps_layer1=32, nr_feature_maps_layer23=32, patch_height=patch_height, patch_width=patch_width)


# encoder32 = init_descr_32(patch_size=16)
# encoder128 = init_descr_128(patch_size=16)


def compute_descriptor(patch, model):
    return model.predict(np.expand_dims(patch, axis=0))[0]#.flatten()


# initialising a descriptor that transforms an image into its intermediate representation,
# and a descriptor that transforms an patch from intermediate representation into the patch descriptor
def init_IR(image_height, image_width, patch_size, model_version=32, nr_feature_maps_layer1=16, nr_feature_maps_layer23=8):

    encoder_trained = load_model(models_dir + '/encoder_' + str(model_version) + '.h5')

    input_shape_img = (image_height, image_width, nr_channels)

    input_img = Input(shape=input_shape_img)

    x = Conv2D(nr_feature_maps_layer1, (3, 3), activation="relu", padding="same")(input_img)
    x = Conv2D(nr_feature_maps_layer23, (3, 3), activation="relu", padding="same")(x)
    encoded_IR = Conv2D(nr_feature_maps_layer23, (3, 3), activation="relu", padding="same")(x)

    encoder_IR = Model(input_img, encoded_IR)

    for i in range(len(encoder_IR.layers)):
        encoder_IR.get_layer(index=i).set_weights(encoder_trained.get_layer(index=i).get_weights())


    input_shape_IR = (patch_size, patch_size, nr_feature_maps_layer23)
    input_IR = Input(shape=input_shape_IR)

    encoded_mp = MaxPooling2D((8, 8), padding="same")(input_IR)

    encoder_mp = Model(input_IR, encoded_mp)

    return encoder_IR, encoder_mp


# the only two parameter options available (i.e. trained)
# -- the two versions of descriptor using 32 and 128 units for the code layer

def init_IR_32(image_height, image_width, patch_size):
    return init_IR(image_height, image_width, patch_size, model_version=32, nr_feature_maps_layer1=16, nr_feature_maps_layer23=8)


def init_IR_128(image_height, image_width, patch_size):
    return init_IR(image_height, image_width, patch_size, model_version=128, nr_feature_maps_layer1=32, nr_feature_maps_layer23=32)


# compute intermediate representation of an image
def compute_IR(image, model):
    return model.predict(np.expand_dims(image, axis=0))[0]


# compute a patch descriptor from an intermediate representation of an image
def compute_descriptor_from_IR(ir, x_coord, y_coord, patch_size, model):
    patch = ir[x_coord : x_coord + patch_size, y_coord : y_coord + patch_size, :]
    return model.predict(np.expand_dims(patch, axis=0))[0]#.flatten()

