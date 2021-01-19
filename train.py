""" Python script to train the two-output classifier system
ATTEMPT:
    To use ImageDataGenerator to train.
"""

# Necessary imports

from colourAndClothingNet import colourAndClothingNet as ccnet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import pandas as pd
import random
import numpy as np
import pickle
from cv2 import cv2
import time

# Defining the required paths

basePath = '/home/workboots/Datasets/ColorsAndClothes/Data/'
colourPicklePath = '/home/workboots/Projects/MultiOutputImageClassifier/colourEncoder'
clothingPicklePath = '/home/workboots/Projects/MultiOutputImageClassifier/clothingPicklePath'
historyPicklePath = '/home/workboots/Projects/MultiOutputImageClassifier/history'
modelPath = '/home/workboots/Projects/MultiOutputImageClassifier/ccnetmodel.h5'

# Getting the filepaths for the data into a list to use for generator-based fitting

imgPaths = sorted([basePath + filename for filename in os.listdir(basePath)])
random.seed(32)  # Shuffling the data
random.shuffle(imgPaths)

# Getting the number of elements that exist

imgNum = len(imgPaths)
print("[INFO] Number of images: {}".format(imgNum))

# Getting the colours and clothing classes for the data

targetColour = np.empty(shape=(imgNum, 1), dtype=str)
targetClothing = np.empty(shape=(imgNum, 1), dtype=str)

for i, imagePath in enumerate(imgPaths):
    classes = imagePath.split(os.path.sep)[-1].split("_")
    targetColour[i] = classes[0]
    targetClothing[i] = classes[1]


# One-hot encoding the colours and classes
# For colours:
ohecolourobj = OneHotEncoder(handle_unknown='ignore')
colourCoded = ohecolourobj.fit_transform(
    targetColour.reshape(-1, 1))

# For clothing:
oheclothingobj = OneHotEncoder(handle_unknown='ignore')
clothingCoded = oheclothingobj.fit_transform(
    targetClothing.reshape(-1, 1))

# Printing the categories for each one hot encoder
colourNum = len(ohecolourobj.categories_[0])
print("[INFO] Number of categories for colour: {}".format(colourNum))

clothingNum = len(oheclothingobj.categories_[0])
print("[INFO] Number of categories for clothing: {}".format(clothingNum))
# Pickling the OneHotEncoder objects to utilize later

with open(colourPicklePath, 'wb') as f:
    pickle.dump(colourCoded, f)

with open(clothingPicklePath, 'wb') as f:
    pickle.dump(clothingCoded, f)

# Defining a Pandas dataframe to store the file paths and their classes for file generation

df = pd.DataFrame({
    'path': imgPaths,
    'colour': pd.Series(list(colourCoded.toarray())),
    'clothing': pd.Series(list(clothingCoded.toarray()))})

# Checking the memory utilization by the metadata dataframe

memoryUsage = (df.memory_usage(index=True, deep=True).sum())/(1024.0 * 1000)
print("[INFO] Memory used by metadata Pandas dataframe: {:.2f}MB".format(
    memoryUsage))

# For visualizing the dataframe

#  print("[INFO] First few data entries in the metadata dataframe are:")
#  print(df.head())

# Checking the train and test splits
#  print("[INFO] The training split dataframe is as follows:")
#  print(train_df.head())

#  print("[INFO] The testing split dataframe is as follows:")
#  print(test_df.head())


def batch_generator_train(dataframe, size, channels, batch_size, categoryNums, is_train,):
    """Used to generate batches of data to be utilized by the fit_generator method. Takes the metadata dataframe and generates batches by fetching the images. The parameters are:
    dataframe: The metadata dataframe to be passed to generate batches.
    size: tuple (height, width) for resizing the images
    channels: The number of channels (e.g. 3 for green, 1 for grayscale)
    batch_size: The number of images in a batch.
    categoryNums: A dictionary containing the number of categories for each classification type.
    is_train: Whether it is training or not."""

    df = dataframe

    # Defining a starting index and finding number of iterations for an epoch with a given batch size
    #  print("[INFO] Number of images: {}".format(len(dataframe.index)))

    # Getting the exact number of elements to be used
    roundedsize = (len(df.index)//batch_size) * batch_size

    #  print("[INFO] Number of iterations per epoch: {}".format(iterations))

    # Defining numpy arrays that will store the image data, the colour data and the clothing data from the dataframe
    batchData = np.zeros(
        (batch_size, size[0], size[1], channels), dtype='float32')

    colourCode = np.zeros(
        (batch_size, categoryNums['colour']), dtype='int32')

    clothingCode = np.zeros(
        (batch_size, categoryNums['clothing']), dtype='int32')

    # The reason for creating this outside is to prevent constant memory allocation and reutilization of the memory

    # Creating a loop to loop through the iterations

    idx = 0

    while True:
        # Clearing out any previous data if any
        batchData.fill(0)
        colourCode.fill(0)
        clothingCode.fill(0)

        #  print("\n[INFO] Iteration: {}/{}".format(iter, iterations))

        #  start = time.time()
        for row in range(batch_size):

            # Reseting the index after every pass through
            idx = idx % roundedsize

            # Shuffling data after full run through
            if(idx == 0):
                #  print("\n[INFO][TRAIN] Shuffling.\n")
                df = df.sample(frac=1).reset_index(drop=True)

            # Reading the image from the given path
            img = cv2.imread(df['path'][idx])
            img = cv2.resize(img, size)  # Resizing to the desired size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Changing BGR to RGB
            img = img_to_array(img)  # Converting to a numpy array
            batchData[row, :, :, :] = img  # Adding to the batch data

            # Adding to the batch colour target
            colourCode[row, :] = df['colour'][idx]
            # Adding to the batch clothing target
            clothingCode[row, :] = df['clothing'][idx]

            idx = idx + 1  # Incrementing to get a continuous flow of different images

        # Getting the time stats per batch
        #  stop = time.time()
        #  print("[INFO] Time taken: {}s".format(stop - start))

        # Scaling data to be in range 0 to 1
        batchData = batchData/255.
        #  print(idx)

        yield(batchData, {"category_output": clothingCode, "colour_output": colourCode})


def batch_generator_test(dataframe, size, channels, batch_size, categoryNums, is_train,):
    """Used to generate batches of data to be utilized by the fit_generator method. Takes the metadata dataframe and generates batches by fetching the images. The parameters are:
    dataframe: The metadata dataframe to be passed to generate batches.
    size: tuple (height, width) for resizing the images
    channels: The number of channels (e.g. 3 for green, 1 for grayscale)
    batch_size: The number of images in a batch.
    categoryNums: A dictionary containing the number of categories for each classification type.
    is_train: Whether it is training or not."""

    df = dataframe

    # Defining a starting index and finding number of iterations for an epoch with a given batch size
    #  print("[INFO] Number of images: {}".format(len(dataframe.index)))

    # Getting the exact number of elements to be used
    roundedsize = (len(df.index)//batch_size) * batch_size

    #  print("[INFO] Number of iterations per epoch: {}".format(iterations))

    # Defining numpy arrays that will store the image data, the colour data and the clothing data from the dataframe
    batchData = np.zeros(
        (batch_size, size[0], size[1], channels), dtype='float32')

    colourCode = np.zeros(
        (batch_size, categoryNums['colour']), dtype='int32')

    clothingCode = np.zeros(
        (batch_size, categoryNums['clothing']), dtype='int32')

    # The reason for creating this outside is to prevent constant memory allocation and reutilization of the memory

    # Creating a loop to loop through the iterations

    idx = 0

    while True:
        # Clearing out any previous data if any
        batchData.fill(0)
        colourCode.fill(0)
        clothingCode.fill(0)

        #  print("\n[INFO] Iteration: {}/{}".format(iter, iterations))

        #  start = time.time()
        for row in range(batch_size):

            # Reseting the index after every pass through
            idx = idx % roundedsize

            # Shuffling data after full run through
            if(idx == 0):
                #  print("\n[TEST][TRAIN] Shuffling.\n")
                df = df.sample(frac=1).reset_index(drop=True)

            # Reading the image from the given path
            img = cv2.imread(df['path'][idx])
            img = cv2.resize(img, size)  # Resizing to the desired size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Changing BGR to RGB
            img = img_to_array(img)  # Converting to a numpy array
            batchData[row, :, :, :] = img  # Adding to the batch data

            # Adding to the batch colour target
            colourCode[row, :] = df['colour'][idx]
            # Adding to the batch clothing target
            clothingCode[row, :] = df['clothing'][idx]

            idx = idx + 1  # Incrementing to get a continuous flow of different images

        # Getting the time stats per batch
        #  stop = time.time()
        #  print("[INFO] Time taken: {}s".format(stop - start))

        # Scaling data to be in range 0 to 1
        batchData = batchData/255.
        #  print(idx)

        yield(batchData, {"category_output": clothingCode, "colour_output": colourCode})


# Creating a dictionary with the number of classes for each classification type
categoryNums = {'colour': colourNum, 'clothing': clothingNum}

# Creating train and test splits

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=32, shuffle=True)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Viewing the metadata dataframes

#  print(train_df.head())

#  print(test_df.head())

# Building the model

model = ccnet.build(96, 96,
                    categoryNums['clothing'],
                    categoryNums['colour'],
                    finalActivation='softmax')

# Defining losses

losses = {
    "category_output": "categorical_crossentropy",
    "colour_output": "categorical_crossentropy"}

# Emphasizing the loss penalization for clothing categorization as it is
# a more difficult task
lossWeights = {
    "category_output": 3.0,
    "colour_output": 1.0}

# Defining a callback to reduce the LR to escape plateaus based on the val_cateogry_output_accuracy attribute
clothingLRAdjust = ReduceLROnPlateau(monitor="val_category_output_accuracy",
                                     factor=0.8,
                                     patience=3,
                                     min_delta=0.04,
                                     verbose=1)

print("[INFO] Compiling model.")

# Defining optimizers. metrics and epochs
init_lr = 1e-2
epochs = 50
optimizer = Adam(lr=init_lr, decay=init_lr/epochs)
training_size = len(train_df.index)
batch_size = 32
steps_per_epoch = (training_size//batch_size) - 1
validation_size = len(test_df)
val_batch_size = 32
val_steps_per_epoch = (validation_size//val_batch_size) - 1

print("[INFO] Attributes: \nepochs={}\ntrainingsize={}\nsteps_per_epoch={}\nbatch_size={}\nvalidationsize={}\nvalidation_steps_per_epoch={}\nvalidation_batch_size={}".format(
    epochs, training_size, steps_per_epoch, batch_size, validation_size, val_steps_per_epoch, val_batch_size))

# Compiling the model

model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=lossWeights,
    metrics=["accuracy"])

# Fitting the model with the custom generators

history = model.fit(
    batch_generator_train(train_df, size=(96, 96), channels=3,
                          batch_size=batch_size, categoryNums=categoryNums, is_train=True),
    validation_data=batch_generator_test(test_df, size=(96, 96), channels=3,
                                         batch_size=val_batch_size, categoryNums=categoryNums, is_train=False),
    verbose=1, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps_per_epoch, validation_batch_size=val_batch_size,
    callbacks=[clothingLRAdjust])

print("[INFO] Training complete. Saving model and training history.")

# Saving model details

model.save(modelPath)

# Saving the history data to use for plotting losses and accuracies later

with open(historyPicklePath, 'wb') as f:
    pickle.dump(history.history, f)

print("[INFO] Complete.")
