import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPool1D, Flatten, concatenate, UpSampling1D, UpSampling2D
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout, Add, LeakyReLU, GaussianNoise,PReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy


def ecg_sr_model(inp_shape,sampling_ratio):
    '''
    Super Resolution Model for ECG Dataset
    :param inp_shape: Tuple(int)
    :param sampling_ratio: int
    :return: Keras Model
    '''
    inp = Input(shape=inp_shape)
    n = Conv1D(32, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')(inp)
    temp = n

    for i in range(4):  # Number of residual blocks
        nn = Conv1D(32, 3, 1, padding='same', kernel_initializer='he_normal')(n)
        nn = BatchNormalization()(nn)
        nn = PReLU()(nn)
        nn = Conv1D(32, 3, 1, padding='same', kernel_initializer='he_normal')(nn)
        nn = BatchNormalization()(nn)
        nn = Add()([n, nn])
        n = nn

    n = Conv1D(32, 3, 1, padding='same', kernel_initializer='he_normal')(n)
    n = BatchNormalization()(n)
    n = Add()([n, temp])

    n_upsample = int(np.log2(sampling_ratio))
    for i in range(n_upsample):
        n = Conv1D(64, 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = UpSampling1D(size=2)(n)
        n = Conv1D(64, 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = PReLU()(n)

    n = Conv1D(1, 1, 1, padding='same', kernel_initializer='he_normal')(n)
    gen = Model(inputs=inp, outputs=n, name='SR_generator')
    return gen

def ecg_imp_model(inp_shape):
    '''
    Imputation model for ECG Dataset
    :param inp_shape: Tuple(int)
    :return: Keras Model
    '''
    inp = Input(shape=inp_shape)
    outfilters = [32, 64, 128]
    filters = 16
    n = Conv1D(filters, 3, 1, padding='same', kernel_initializer='he_normal')(inp)
    n = BatchNormalization()(n)
    n = PReLU()(n)
    down_array = [n]

    for i in range(len(outfilters)):
        n = Conv1D(outfilters[i], 3, 2, padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = PReLU()(n)
        n = Conv1D(outfilters[i], 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = PReLU()(n)
        down_array.append(n)

    outfilters.reverse()
    outfilters.append(filters)

    for i in range(1,len(outfilters)):
        n = UpSampling1D(size=2)(n)
        n = Conv1D(outfilters[i], 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = PReLU()(n)
        n = concatenate([n,down_array[len(outfilters)-i-1]],axis=-1)
        n = Conv1D(outfilters[i], 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = PReLU()(n)

    n = Conv1D(1,1,1,padding='same', kernel_initializer='he_normal')(n)
    gen = Model(inputs=inp, outputs=n, name='Imp_Generator')
    return gen

def ecg_disc_model(inp_shape):
    '''
    Discriminator Model for ECG Dataset
    :param inp_shape: Tuple(int)
    :return: Keras Model
    '''
    inp = Input(shape=inp_shape)
    outfilters = [16,32]
    filters = 16
    n = Conv1D(filters, 3, 1, padding='same', kernel_initializer='he_normal')(inp)
    n = PReLU()(n)
    n = Conv1D(filters, 3, 2, padding='same', kernel_initializer='he_normal')(n)
    # n = BatchNormalization()(n)
    n = PReLU()(n)

    for i in range(len(outfilters)):
        n = Conv1D(outfilters[i], 3, 1, padding='same', kernel_initializer='he_normal')(n)
        # n = BatchNormalization()(n)
        n = PReLU()(n)
        n = Conv1D(outfilters[i], 3, 2, padding='same', kernel_initializer='he_normal')(n)
        # n = BatchNormalization()(n)
        n = PReLU()(n)

    n = Flatten()(n)
    # n = Dense(256)(n)
    # n = PReLU()(n)
    n = Dense(1)(n)
    # n = Activation('sigmoid')(n)

    disc = Model(inputs=inp, outputs=n, name='Discriminator')
    return disc

def ecg_clf_model(inp_shape,nclasses):
    '''
    Classification Model for ECG Dataset
    :param inp_shape: Tuple(int)
    :param nclasses: Number of classes for classification
    :return: Keras Model
    '''
    inp = Input(shape=inp_shape)
    outfilters = [16, 32, 64]
    filters = 16
    input_length = inp_shape[0]
    n = Conv1D(filters, 3, 1, padding='same', kernel_initializer='he_normal')(inp)
    n = PReLU()(n)
    n = Conv1D(filters, 3, 2, padding='same', kernel_initializer='he_normal')(n)
    n = BatchNormalization()(n)
    n = PReLU()(n)
    input_length/=2

    for i in range(len(outfilters)):
        n = Conv1D(outfilters[i], 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = PReLU()(n)
        if input_length%2==0:
            n = Conv1D(outfilters[i], 3, 2, padding='same', kernel_initializer='he_normal')(n)
            input_length/=2
        else:
            n= Conv1D(outfilters[i], 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = PReLU()(n)

    n = Flatten()(n)
    n = Dense(256)(n)
    n = PReLU()(n)
    n = Dense(nclasses)(n)

    clf = Model(inputs=inp, outputs=n, name='Classifier')
    return clf

def shl_clf_model(inp_shape,nclasses):
    '''
    Classification Model for SHL Dataset
    :param inp_shape: Tuple(int)
    :param nclasses: Number of classes for classification
    :return: Keras Model
    '''
    inp = Input(shape=inp_shape)
    outfilters = [16, 32, 64]
    filters = 16
    input_length = inp_shape[1]
    n = Conv2D(filters, (1,3), (1,1), padding='same', kernel_initializer='he_normal')(inp)
    n = PReLU()(n)
    n = Conv2D(filters, (1,3), (1,2), padding='same', kernel_initializer='he_normal')(n)
    n = BatchNormalization()(n)
    n = PReLU()(n)
    input_length /= 2

    for i in range(len(outfilters)):
        n = Conv2D(outfilters[i], (1,3), (1,1), padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = PReLU()(n)
        if input_length % 2 == 0:
            n = Conv2D(outfilters[i], (1,3), (1,2), padding='same', kernel_initializer='he_normal')(n)
            input_length /= 2
        else:
            n = Conv2D(outfilters[i], (1,3), (1,2), padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = PReLU()(n)

    n = Flatten()(n)
    n = Dense(256)(n)
    n = PReLU()(n)
    n = Dense(nclasses)(n)

    clf = Model(inputs=inp, outputs=n, name='Classifier')
    return clf

def sr_model_func(data_type):
    '''
    Returns super resolution model architecture for the given data type
    :param data_type: str
    :return: Function that returns Keras Model
    '''
    if data_type=='ecg':
        return ecg_sr_model

def imp_model_func(data_type):
    '''
    Returns imputation model architecture for the given data type
    :param data_type: str
    :return: Function that returns Keras Model
    '''
    if data_type=='ecg':
        return ecg_imp_model

def disc_model_func(data_type):
    '''
    Returns discriminator model architecture for the given data type
    :param data_type: str
    :return: Function that returns Keras Model
    '''
    if data_type=='ecg':
        return ecg_disc_model


def clf_model_func(data_type):
    '''
    Returns discriminator model architecture for the given data type
    :param data_type: str
    :return: Function that returns Keras Model
    '''
    if data_type=='ecg':
        return ecg_clf_model

    elif data_type=='shl':
        return shl_clf_model




