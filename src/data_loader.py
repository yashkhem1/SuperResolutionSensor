import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model
import multiprocessing

import tensorflow as tf

def get_class_weights(data_type):
    '''
    Returns the class weights for weighted cross entropy loss
    :param data_type: str
    :return: List[float]
    '''
    if data_type == 'ecg':
        train_data = np.array(pd.read_csv('data/mitbih_train.csv'))
        w0 = len(train_data) / len(train_data[train_data[:, -1] == 0])
        w1 = len(train_data) / len(train_data[train_data[:, -1] == 1])
        w2 = len(train_data) / len(train_data[train_data[:, -1] == 2])
        w3 = len(train_data) / len(train_data[train_data[:, -1] == 3])
        w4 = len(train_data) / len(train_data[train_data[:, -1] == 4])
        weight_list = np.array([w0,w1,w2,w3,w4])
        weight_list = weight_list/weight_list.sum()
        return weight_list



def read_train_data(data_type,rs=False):
    '''
    Read the train and test data for the corresponding data type
    :param data_type: str
    :return: (trainX, trainY)
    '''
    if data_type == 'ecg':
        # read data from csv files
        train_data = pd.read_csv('data/mitbih_train.csv',header=None)
        if rs:
            data_0 = train_data[train_data[187] == 0].sample(n=20000,random_state=122)
            data_1 = train_data[train_data[187] == 1]
            data_2 = train_data[train_data[187] == 2]
            data_3 = train_data[train_data[187] == 3]
            data_4 = train_data[train_data[187] == 4]

            data_1_upsample = resample(data_1, replace=True, n_samples=20000, random_state=123)
            data_2_upsample = resample(data_2, replace=True, n_samples=20000, random_state=124)
            data_3_upsample = resample(data_3, replace=True, n_samples=20000, random_state=125)
            data_4_upsample = resample(data_4, replace=True, n_samples=20000, random_state=126)

            train_data = pd.concat([data_0,data_1_upsample,data_2_upsample,data_3_upsample,data_4_upsample])

        train_data = np.array(train_data)
        np.random.shuffle(train_data)
        train_X =train_data[:, :-1]
        train_Y = train_data[:, -1]
        train_Y = tf.keras.utils.to_categorical(train_Y,num_classes=5)

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
        test_data = pd.read_csv('data/mitbih_test.csv',header=None)
        test_X = np.array(test_data)[:, :-1]
        test_Y = np.array(test_data)[:, -1]
        test_Y = tf.keras.utils.to_categorical(test_Y, num_classes=5)

        # converting the X data to 1 channel data
        test_X = test_X.reshape(-1, test_X.shape[1], 1)

        # padding the X data so that it is of length 192 (divisible by 2,4,8,16)
        test_X = np.pad(test_X, ((0, 0), (2, 3), (0, 0)), 'constant')

        return test_X,test_Y

def train_cf_dataset(data_type,sampling_ratio,batch_size,shuffle_buffer_size=1000,fetch_buffer_size=2,resample=False, sr_model= None,
                     prob=0,seed=1,cont=False,imp_model=None):
    '''
    Returns the train dataset for classification model
    :param data_type: str
    :param sampling_ratio: int
    :param batch_size: int
    :param shuffle_buffer_size: int
    :param buffer_size: int
    :param resample: bool
    :return: Tensorflow Dataset
    '''
    train_X,train_Y = read_train_data(data_type,resample)
    #downsample the high resolution data
    if data_type == 'ecg':
        if sampling_ratio!=1:
            train_X = train_X[:, ::sampling_ratio, :]
            if sr_model:
                G = load_model(sr_model)
                train_X = G.predict(train_X,batch_size=batch_size,verbose=1)
                print(len(train_X))

        if  prob!=0:
            np.random.seed(seed)
            indices = np.arange(192)
            n_missing = int(prob * 192)
            train_X_m = np.zeros(train_X.shape)
            train_mask = np.ones(train_X.shape)
            for i, data in enumerate(train_X):
                if cont:
                    missing_start = np.random.randint(0, int((1 - prob) * 192) + 1)
                    missing_indices = np.arange(missing_start, missing_start + n_missing)
                else:
                    missing_indices = np.random.choice(indices, n_missing, replace=False)
                train_X_m[i] = data
                train_X_m[i][missing_indices] = 0
                train_mask[i][missing_indices] = 0

            if imp_model:
                G = load_model(imp_model)
                train_X_m_mask = np.concatenate([train_X_m,train_mask],axis=-1)
                x_pred = G.predict(train_X_m_mask,batch_size=batch_size,verbose=1)
                train_X = train_X_m*train_mask + x_pred*(1-train_mask)

            else:
                train_X = train_X_m


    #defining the generator to generate dataset
    def generator():
        for i in range(len(train_X)):
            yield train_X[i],train_Y[i]

    train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=fetch_buffer_size)
    train_ds = train_ds.batch(batch_size)
    return train_ds

def test_cf_dataset(data_type,sampling_ratio,batch_size,fetch_buffer_size=2, sr_model = None, prob=0, seed=1, cont=False,
                    imp_model = None):
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
    if data_type == 'ecg':
        if sampling_ratio!=1:
            test_X = test_X[:, ::sampling_ratio, :]
            if sr_model:
                G = load_model(sr_model)
                test_X = G.predict(test_X,batch_size=batch_size,verbose=1)
                print(len(test_X))

        if prob != 0:
            np.random.seed(seed)
            indices = np.arange(192)
            n_missing = int(prob * 192)
            test_X_m = np.zeros(test_X.shape)
            test_mask = np.ones(test_X.shape)
            for i, data in enumerate(test_X):
                if cont:
                    missing_start = np.random.randint(0, int((1 - prob) * 192) + 1)
                    missing_indices = np.arange(missing_start, missing_start + n_missing)
                else:
                    missing_indices = np.random.choice(indices, n_missing, replace=False)
                test_X_m[i] = data
                test_X_m[i][missing_indices] = 0
                test_mask[i][missing_indices] = 0

            if imp_model:
                G = load_model(imp_model)
                test_X_m_mask = np.concatenate([test_X_m, test_mask], axis=-1)
                x_pred = G.predict(test_X_m_mask, batch_size=batch_size, verbose=1)
                test_X = test_X_m * test_mask + x_pred * (1 - test_mask)

            else:
                test_X = test_X_m



    # defining the generator to generate dataset
    def generator():
        for i in range(len(test_X)):
            yield test_X[i], test_Y[i]

    test_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))
    test_ds = test_ds.prefetch(buffer_size=fetch_buffer_size)
    test_ds = test_ds.batch(batch_size)
    return test_ds

def train_sr_dataset(data_type,sampling_ratio,batch_size,shuffle_buffer_size=1000,fetch_buffer_size=2,resample=False):
    '''
    Returns train dataset for super resolution model for the given sampling ratio
    :param data_type: str
    :param sampling_ratio: int
    :param batch_size: int
    :param shuffle_buffer_size: int
    :param fetch_buffer_size: int
    :param resample: bool
    :return: Tensorflow Dataset
    '''
    train_X, train_Y = read_train_data(data_type,resample)
    #downsampling the high resolution dataset
    if data_type == 'ecg':
        train_X_r = train_X[:, ::sampling_ratio, :]

    # defining the generator to generate dataset
    def generator():
        for i in range(len(train_X)):
            yield train_X_r[i], train_X[i], train_Y[i]

    train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32,tf.float32))
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
    test_X, test_Y = read_test_data(data_type)
    # downsampling the high resolution dataset
    if data_type=='ecg':
        test_X_r = test_X[:, ::sampling_ratio, :]

    # defining the generator to generate dataset
    def generator():
        for i in range(len(test_X)):
            yield test_X_r[i], test_X[i], test_Y[i]

    test_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32,tf.float32))
    test_ds = test_ds.prefetch(buffer_size=fetch_buffer_size)
    test_ds = test_ds.batch(batch_size)
    return test_ds

def train_imp_dataset(data_type,batch_size, prob, seed, cont=False, fixed=False, shuffle_buffer_size=1000,fetch_buffer_size=2, resample=False):
    '''
    Returns train dataset for missing data imputation model for the given sampling ratio
    :param data_type: str
    :param sampling_ratio: int
    :param batch_size: int
    :param prob: float
    :param seed: int
    :param shuffle_buffer_size: int
    :param fetch_buffer_size: int
    :param resample: bool
    :return: Tensorflow Dataset
    '''
    train_X, train_Y = read_train_data(data_type,resample)
    np.random.seed(seed)
    #downsampling the high resolution dataset
    if data_type == 'ecg':
        indices = np.arange(192)
        n_missing = int(prob*192)
        if fixed:
            train_X_m = np.zeros(train_X.shape)
            train_mask = np.ones(train_X.shape)
            for i,data in enumerate(train_X):
                if cont:
                    missing_start =np.random.randint(0,int((1-prob)*192)+1)
                    missing_indices = np.arange(missing_start,missing_start+n_missing)
                else:
                    missing_indices = np.random.choice(indices,n_missing,replace=False)
                train_X_m[i] = data
                train_X_m[i][missing_indices]=0
                train_mask[i][missing_indices]=0
            print("Loaded Training Data")


    # defining the generator to generate dataset
    def generator():
        for i in range(len(train_X)):
            if fixed:
                yield train_X_m[i], train_mask[i],train_X[i], train_Y[i]
            else:
                yield train_X[i],train_Y[i]


    def map_function_ecg(train_X,train_Y):
        sample = train_X
        train_mask_sample = np.ones(sample.shape)
        if cont:
            missing_start = np.random.randint(0, int((1 - prob) * 192) + 1)
            missing_indices = np.arange(missing_start, missing_start + n_missing)
        else:
            missing_indices = np.random.choice(indices, n_missing, replace=False)
        train_X_m_sample = sample.copy()
        train_X_m_sample[missing_indices] = 0
        train_mask_sample[missing_indices] = 0
        yield train_X_m_sample, train_mask_sample, train_X, train_Y

    if not fixed:
        if data_type=='ecg':
            train_ds =tf.data.Dataset.from_generator(generator, output_types=(tf.float32,tf.float32))
            train_ds = train_ds.map(map_function_ecg,num_parallel_calls=multiprocessing.cpu_count())
    else:
        train_ds = tf.data.Dataset.from_generator(generator,
                                                  output_types=(tf.float32, tf.float32, tf.float32, tf.float32))
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=fetch_buffer_size)
    train_ds = train_ds.batch(batch_size)
    return train_ds

def test_imp_dataset(data_type,batch_size,prob,seed,cont=False,fetch_buffer_size=2):
    '''
    Returns test dataset for super resolution model for the given sampling ratio
    :param data_type: str
    :param sampling_ratio: int
    :param batch_size: int
    :param fetch_buffer_size: int
    :return: Tensorflow Dataset
    '''
    test_X, test_Y = read_test_data(data_type)
    # downsampling the high resolution dataset
    if data_type=='ecg':
        np.random.seed(seed)
        indices = np.arange(192)
        n_missing = int(prob * 192)
        test_X_m = np.zeros(test_X.shape)
        test_mask = np.ones(test_X.shape)
        for i, data in enumerate(test_X):
            if cont:
                missing_start = np.random.randint(0, int((1 - prob) * 192) + 1)
                missing_indices = np.arange(missing_start, missing_start + n_missing)
            else:
                missing_indices = np.random.choice(indices, n_missing, replace=False)
            test_X_m[i] = data
            test_X_m[i][missing_indices] = 0
            test_mask[i][missing_indices] = 0

    print("Loaded Testing Data")

    # defining the generator to generate dataset
    def generator():
        for i in range(len(test_X)):
            yield test_X_m[i], test_mask[i], test_X[i], test_Y[i]

    test_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32, tf.float32,tf.float32))
    test_ds = test_ds.prefetch(buffer_size=fetch_buffer_size)
    test_ds = test_ds.batch(batch_size)
    return test_ds






