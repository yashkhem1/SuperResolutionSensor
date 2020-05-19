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

def read_train_data(data_type):
    '''
    Read the train and test data for the corresponding data type
    :param data_type: str
    :return: (trainX, trainY, testX, testY)
    '''
    if data_type == 'ecg':
        # read data from csv files
        train_data = pd.read_csv('../data/mitbih_train.csv')
        np.random.shuffle(train_data)
        train_X =np.array(train_data)[:, :-1]
        train_Y = np.array(train_data)[:, -1].astype(int)

        # converting the X data to 1 channel data
        train_X = train_X.reshape(-1, train_X.shape[1], 1)

        # padding the X data so that it is of length 192 (divisible by 2,4,8,16)
        train_X = np.pad(train_X, ((0, 0), (2, 3), (0, 0)), 'constant')

        return train_X,train_Y

def read_test_data(data_type):
    '''
    Read the test data for the corresponding data type
    :param data_type: str
    :return: (testX,testY)
    '''

    if data_type=='ecg':
        # read data from csv files
        test_data = pd.read_csv('../data/mitbih_test.csv')
        test_X = np.array(test_data)[:, :-1]
        test_Y = np.array(test_data)[:, -1].astype(int)

        # converting the X data to 1 channel data
        test_X = test_X.reshape(-1, test_X.shape[1], 1)

        # padding the X data so that it is of length 192 (divisible by 2,4,8,16)
        test_X = np.pad(test_X, ((0, 0), (2, 3), (0, 0)), 'constant')

        return test_X,test_Y

def train_cf_dataset(data_type,sampling_ratio,batch_size,shuffle_buffer_size=1000,fetch_buffer_size=2):
    '''
    Returns the train dataset for classification model
    :param data_type: str
    :param sampling_ratio: int
    :param batch_size: int
    :param shuffle_buffer_size: int
    :param buffer_size: int
    :return: Tensorflow Dataset
    '''
    train_X,train_Y = read_train_data(data_type)
    #downsample the high resolution data
    train_X = train_X[:, ::sampling_ratio, :]

    #defining the generator to generate dataset
    def generator():
        for i in range(len(train_X)):
            yield train_X[i],train_Y[i]

    train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=fetch_buffer_size)
    train_ds = train_ds.batch(batch_size)
    return train_ds

def test_cf_dataset(data_type,sampling_ratio,batch_size,fetch_buffer_size=2):
    '''
    Returns the test dataloader for classification model
    :param data_type: str
    :param sampling_ratio: int
    :param batch_size: int
    :param fetch_buffer_size: int
    :return: Tensorflow Dataset
    '''
    test_X, test_Y = read_test_data(data_type)
    #downsample the high resolution data
    test_X = test_X[:, ::sampling_ratio, :]

    # defining the generator to generate dataset
    def generator():
        for i in range(len(test_X)):
            yield test_X[i], test_Y[i]

    test_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
    test_ds = test_ds.prefetch(buffer_size=fetch_buffer_size)
    test_ds = test_ds.batch(batch_size)
    return test_ds

def train_sr_dataset(data_type,sampling_ratio,batch_size,shuffle_buffer_size=1000,fetch_buffer_size=2):
    '''
    Returns train dataset for super resolution model for the given sampling ratio
    :param data_type: str
    :param sampling_ratio: int
    :param batch_size: int
    :param shuffle_buffer_size: int
    :param fetch_buffer_size: int
    :return: Tensorflow Dataset
    '''
    train_X, _ = read_train_data(data_type)
    #downsampling the high resolution dataset
    train_X_r = train_X[:, ::sampling_ratio, :]

    # defining the generator to generate dataset
    def generator():
        for i in range(len(train_X)):
            yield train_X_r[i], train_X[i]

    train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=fetch_buffer_size)
    train_ds = train_ds.batch(batch_size)
    return train_ds


def test_sr_dataset(data_type,sampling_ratio,batch_size,fetch_buffer_size=2):
    '''
    Returns test dataset for super resolution model for the given sampling ratio
    :param data_type: str
    :param sampling_ratio: int
    :param batch_size: int
    :param fetch_buffer_size: int
    :return: Tensorflow Dataset
    '''
    test_X,_ = read_test_data(data_type)
    # downsampling the high resolution dataset
    test_X_r = test_X[:, ::sampling_ratio, :]

    # defining the generator to generate dataset
    def generator():
        for i in range(len(test_X)):
            yield test_X_r[i], test_X[i]

    test_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))
    test_ds = test_ds.prefetch(buffer_size=fetch_buffer_size)
    test_ds = test_ds.batch(batch_size)
    return test_ds






