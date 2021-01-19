""" Creates a two output classifier model to classify images by the clothing type and the colour."""


# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Model  # To use the functional API
from tensorflow.keras.layers import Conv2D  # For the basic convolutions
# To pool layers after convolutions
from tensorflow.keras.layers import MaxPooling2D
# Implementing activation as a separate layer to keep the layout looking clean
from tensorflow.keras.layers import Activation
# To convert colour images to grayscale with a lambda function
from tensorflow.keras.layers import Lambda
# To implement dropouts during training
from tensorflow.keras.layers import Dropout
# To utilize in the end of the network branches
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten  # To use for getting the outputs
# For the input layer creation when building the model
from tensorflow.keras.layers import Input
# To carry out batch normalization during training
from tensorflow.keras.layers import BatchNormalization


class colourAndClothingNet:

    @staticmethod
    def build_clothing_branch(inputs, numCategories,
                              finalActivation='softmax', channelDim=-1):
        """Builds the clothing classifier branch of the model. The parameters are:
        inputs: The inputs.
        numCategories: The number of categories the network needs to be able to classify.
        e.g. for being able to classify dresses, shoes, shirts and pants, the value is 4.
        finalActivation: Species the activation for the final layer. This is of use in binary
        vs multi classification where the activations are usually used as sigmoid in binary and
        softmax in multi.
        channelDim: The channel dimension specification for the Batch Normalization layer.
        To understand this more, read up about the BatchNormalization layer in Keras documentation."""
        # 1) Eliminvation colour with a Lambda layer to only utilize structural information
        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        # 2) CONV => RELU => BNORM => MAXPOOL => DROPOUT

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=channelDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # 3) (CONV => RELU => BNORM) * 2 => MAXPOOL => DROPOUT

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=channelDim)(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=channelDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # 4) FL => FC => RELU => BNORM => DROPOUT => FC => SOFTMAX(OUTPUT)

        x = Flatten()(x)
        x = Dense(units=256)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(units=numCategories)(x)
        x = Activation(finalActivation, name="category_output")(x)

        return x

    @staticmethod
    def build_colour_branch(inputs, numColours,
                            finalActivation='softmax', channelDim=-1):
        """Builds the colours classifier branch of the model. The parameters are:
        inputs: The inputs.
        numColours: The number of colours the network needs to be able to classify.
        finalActivation: Species the activation for the final layer. This is of use in binary
        vs multi classification where the activations are usually used as sigmoid in binary and
        softmax in multi.
        channelDim: The channel dimension specification for the Batch Normalization layer.
        To understand this more, read up about the BatchNormalization layer in Keras documentation."""
        #  1) (CONV => RELU => BNORM => MAXPOOL => DROPOUT) * 3

        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=channelDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=channelDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=channelDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # 2) FL => FC => RELU => BNORM => DROPOUT => FC => SOFTMAX(OUTPUT)

        x = Flatten()(x)
        x = Dense(units=128)(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=channelDim)(x)
        x = Dropout(0.5)(x)
        x = Dense(units=numColours)(x)
        x = Activation(finalActivation, name="colour_output")(x)

        return x

    @staticmethod
    def build(width, height, numCategories,
              numColours, finalActivation='softmax'):
        """Builds the two output classifier model. The parameters are:
        width, height: The image size to be used.
        numCategories: The number of clothing categories(passed to the build_clothing_branch() method).
        numColours: The number of colours(passed to the build_colour_branch() method.
        finalActivation: The activation in the last layer passed to both branches."""

        inputShape = (height, width, 3)
        channelDim = -1

        inputs = Input(shape=inputShape)

        categoryBranch = colourAndClothingNet.build_clothing_branch(
            inputs, numCategories, finalActivation, channelDim)

        colourBranch = colourAndClothingNet.build_colour_branch(
            inputs, numColours, finalActivation, channelDim)

        # Defining the model

        model = Model(
            inputs=inputs,
            outputs=[categoryBranch, colourBranch],
            name="colourandclothingnet")

        return model
