#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import cv2
from random import shuffle
from os import listdir, path
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

train_data_directory = "./data/train"
validation_data_directory = "./data/dev"
IMG_SIZE = 64
CHANNELS = 1
class_size = 2
lr = 0.01


def label_data(img_name):
    """
    :param img_name: The image name
    :return: The image label
    """
    name = str(img_name)[:3]
    if name == "cat":
        return [1, 0]
    elif name == "dog":
        return [0, 1]

def create_data(img_dir, validation_data=False):
    """
    :param img_dir: The images directory
    :param validation_data: Flag to decide if data should be augmented for training
    :return: A list of all the image data
    """
    data = []
    for img in tqdm(listdir(img_dir)):
        label = label_data(img)
        img_path = path.join(img_dir, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        data.append([np.array(img1)/255, np.array(label)])
        if not validation_data:
            """
            Augment the training data by horizontally flipping each training example to 
            generate a "new" training example
            """
            img2 = cv2.flip(img1, 1)
            data.append([np.array(img2)/255, np.array(label)])
    
    shuffle(data)
    return data
    

train_data = create_data(train_data_directory)

validation_data = create_data(validation_data_directory, True)


X_train = np.array([data[0] for data in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)
Y_train = np.array([data[1] for data in train_data]).reshape(-1, class_size)

X_train_mlp = np.array([data[0] for data in train_data]).reshape(-1, IMG_SIZE * IMG_SIZE * CHANNELS)



X_validation = np.array([data[0] for data in validation_data]).reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)
Y_validation = np.array([data[1] for data in validation_data]).reshape(-1, class_size)

X_validation_mlp = np.array([data[0] for data in validation_data]).reshape(-1, IMG_SIZE * IMG_SIZE * CHANNELS)


def mlp_model(input_shape, layer_sizes, dropout_rate=0.0):
    """
    Function to create and return a deep multilayer perceptron model for image classification
    :param input_shape: The shape of the input data
    :param layer_sizes: The number of units in the hidden and output layers
    :param dropout_rate: The dropout rate
    
    :return: An n-Hidden Layer MLP model
    """
    input_layer = tf.keras.Input(shape=input_shape)
    X = tf.keras.layers.Dense(layer_sizes[0])(input_layer)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(dropout_rate)(X)
    
    for layer_size in layer_sizes[1:-1]:
        X = tf.keras.layers.Dense(layer_size)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(dropout_rate)(X)
        
    X = tf.keras.layers.Dense(layer_sizes[-1])(X)
    X = tf.keras.layers.BatchNormalization()(X)
    output = tf.keras.layers.Activation('softmax')(X)
    model = tf.keras.Model(inputs=input_layer, outputs=output)
        
    return model


model = mlp_model((IMG_SIZE * IMG_SIZE * CHANNELS, ), [1500, 1500, 1000, 500, class_size], 0.2)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


history = model.fit(X_train_mlp, Y_train, batch_size=32, epochs=30)


predictions = model.predict(X_validation_mlp, batch_size=32)


correct = tf.equal(tf.math.argmax(predictions, axis=1), tf.math.argmax(Y_validation, axis=1))
validation_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print("Validation Accuracy = ", float(validation_accuracy) * 100, "%")


# model.save("./models/mlp_model.h5")

n_row, n_col = 3, 3
samples = np.random.choice(len(X_validation_mlp), n_row * n_col, replace=False)


_, ax = plt.subplots(n_row, n_col)
for i in range(n_row):
    for j in range(n_col):
        ax[i, j].imshow(X_validation_mlp[samples[i * n_col + j]].reshape(IMG_SIZE, IMG_SIZE))
        ax[i, j].label_outer()
        label = "Cat" if float(tf.math.argmax(predictions[samples[i * n_col + j]])) == 0.0 else "Dog"
        ax[i, j].set_title(label)
plt.show()


def simple_cnn_model(input_shape, n_conv_blocks, n_dense, n_classes, dropout_rate=0.0):
    """
    Function to create an return a convolutional neural network model for image classification
    :param input_shape: The image input shape
    :param n_conv_blocks: The number of convolution-maxpool blocks
    :param n_dense: The number of units in the dense layers
    :param n_classes: The number of output classes
    :param dropout_rate: The dropout rate
    
    :return: A CNN model
    """
    input_layer = tf.keras.Input(shape=input_shape)
    X = input_layer
    
    for i in range(n_conv_blocks):
        X = tf.keras.layers.Conv2D(32 * 2 ** i, 5, strides=(1, 1), padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(dropout_rate)(X)
        X = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
      
    X = tf.keras.layers.Flatten()(X)
    for n in n_dense:
        X = tf.keras.layers.Dense(n)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(dropout_rate)(X)
    
    X = tf.keras.layers.Dense(n_classes)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    output = tf.keras.layers.Activation('softmax')(X)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    return model


cnn_model = simple_cnn_model((IMG_SIZE, IMG_SIZE, 1, ), 3, [500, 500], 2, 0.2)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.9, beta_2=0.999)
cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


cnn_model.summary()


history = cnn_model.fit(X_train, Y_train, batch_size=32, epochs=10)


cnn_predictions = cnn_model.predict(X_validation, batch_size=32)

correct = tf.equal(tf.math.argmax(cnn_predictions, axis=1), tf.math.argmax(Y_validation, axis=1))
validation_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print("CNN Validation Accuracy = ", float(validation_accuracy) * 100, "%")


# cnn_model.save("./models/cnn_model.h5")

n_row, n_col = 3, 3
cnn_samples = np.random.choice(len(X_validation), n_row * n_col, replace=False)


_, ax = plt.subplots(n_row, n_col)
for i in range(n_row):
    for j in range(n_col):
        ax[i, j].imshow(X_validation[cnn_samples[i * n_col + j]].reshape(IMG_SIZE, IMG_SIZE))
        ax[i, j].label_outer()
        label = "Cat" if float(tf.math.argmax(predictions[cnn_samples[i * n_col + j]])) == 0.0 else "Dog"
        ax[i, j].set_title(label)
plt.show()


def residual_block(X, n_conv_layers, n_filters, filter_size, dropout_rate=0.0):
    """
    Function to create and return a residual block for a residual network model
    :param X: The input to the block
    :param n_conv_layers: The number of convolutional layers in the block
    :param n_filters: The number of filters in each convolutional layer
    :param filter_size: The filter size
    :param dropout_rate: The dropout rate

    :return: A residual block
    """
    assert (n_conv_layers >= 2), "A residual block should have at least 2 layers"
    X_copy = X

    if X_copy.shape[-1] != n_filters:
        X_copy = tf.keras.layers.Conv2D(n_filters, filter_size, padding='same', strides=(1, 1))(X_copy)
        X_copy = tf.keras.layers.BatchNormalization()(X_copy)

    for i in range(n_conv_layers):
        X = tf.keras.layers.Conv2D(n_filters, filter_size, padding='same', strides=(1, 1))(X)
        X = tf.keras.layers.BatchNormalization()(X)
        if i < n_conv_layers - 1:
            X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Add()([X_copy, X])
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(dropout_rate)(X)

    return X



def simple_residual_network(input_shape, res_block_sizes, n_dense, n_classes, dropout_rate=0.0):
    """
    Function to create and return a simple residual network model for image classification
    :param input_shape: The input shape
    :param res_block_sizes: A list containing the number of conv layers in each residual block
    :param n_dense: A list containing the number of units in each fully-connected/dense layer
    :param n_classes: The number of output classes
    :param dropout_rate: The dropout rate

    :return: A residual network model
    """
    input_layer = tf.keras.layers.Input(shape=input_shape)
    X = tf.keras.layers.Conv2D(64, 3, padding='same', strides=(1, 1))(input_layer)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = tf.keras.layers.Dropout(dropout_rate)(X)

    i = 1
    for res_block_size in res_block_sizes:
        X = residual_block(X, res_block_size, 64 * 2 ** (i // 2), 3, dropout_rate)
        if i < len(res_block_sizes) - 1:
            X = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)
        i += 1

    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    X = tf.keras.layers.Flatten()(X)
    
    for n in n_dense:
        X = tf.keras.layers.Dense(n)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(dropout_rate)(X)

    X = tf.keras.layers.Dense(n_classes)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    output = tf.keras.layers.Activation('softmax')(X)

    model = tf.keras.Model(inputs=input_layer, outputs=output)

    return model


resnet_model = simple_residual_network((IMG_SIZE, IMG_SIZE, 1, ), [2, 2, 2, 2], [500], 2, 0.2)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1 = 0.9, beta_2=0.999)
resnet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


resnet_model.summary()


history = resnet_model.fit(X_train, Y_train, batch_size=32, epochs=10)

resnet_predictions = resnet_model.predict(X_validation, batch_size=32)

correct = tf.equal(tf.math.argmax(resnet_predictions, axis=1), tf.math.argmax(Y_validation, axis=1))
validation_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print("Resnet Validation Accuracy = ", float(validation_accuracy) * 100, "%")


# resnet_model.save("./models/resnet_model.h5")

n_row, n_col = 3, 3
resnet_samples = np.random.choice(len(X_validation), n_row * n_col, replace=False)


_, ax = plt.subplots(n_row, n_col)
for i in range(n_row):
    for j in range(n_col):
        ax[i, j].imshow(X_validation[resnet_samples[i * n_col + j]].reshape(IMG_SIZE, IMG_SIZE))
        ax[i, j].label_outer()
        label = "Cat" if float(tf.math.argmax(predictions[resnet_samples[i * n_col + j]])) == 0.0 else "Dog"
        ax[i, j].set_xlabel(label)
plt.show()
