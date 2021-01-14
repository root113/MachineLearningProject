import cv2
import numpy as np
import os
import sys
import tensorflow as tf

# from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.optimizers import Adam

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.20
TRAIN_SIZE = 0.70
BATCH_SIZE = 64

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, train_size=TRAIN_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []
    dim = (IMG_WIDTH, IMG_HEIGHT)

    # load all 42 sub-directories in data_dir
    for folder in os.listdir(data_dir):
        # join folder path
        folder_path = os.path.join(data_dir, folder)

        # check to see if path is valid
        if os.path.isdir(folder_path):

            # read image
            for file in os.listdir(folder_path):
                image = cv2.imread(os.path.join(folder_path, file))

                # resize the image
                resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                # append to images and labels list
                images.append(resized_image)
                labels.append(int(folder))

    return images, labels
    # raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # create a convolutional neural network
    model = tf.keras.models.Sequential([
        # convolutional layer. Learn 32 filters using 3x3 kernel
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # pooling layer using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten units
        tf.keras.layers.Flatten(),
        # add hidden layers with dropout
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        # add output layer with output units for all 43 categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax") # softmax turns output to probability distribution
    ])

    # train neural network
    #model.compile(
    #    optimizer="adam",
    #    loss="categorial_crossentropy",
    #    metrics=["accuracy"]
    #)

    # opt = Adam(lr=0.001, decay=0.001 / EPOCHS)
    opt = SGD(lr=0.00101, decay=0.001, momentum=0.9, nesterov=True) # better performance

    # print("Time spent: ",time.time()-t0)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    # train the network
    # print("Time spent: ",time.time()-t0)
    print("\nNetwork is being trained...\n")
    # H = model.fit_generator(
    #     aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    #     validation_data=(testX, testY),
    #     teps_per_epoch=len(trainX) // BATCH_SIZE,
    #     epochs=EPOCHS, verbose=1)

    return model
    # raise NotImplementedError


if __name__ == "__main__":
    main()