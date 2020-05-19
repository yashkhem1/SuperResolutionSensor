# -*- coding: utf-8 -*-
"""SuperResolution.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sJw4eWc6buP_WZQ1VAvfbjqEyer0yfN9
"""

"""In this notebook we are going to reconstruct high sampling rate ECG data from low sampling rate using GAN-based methods. We will use MIT-BIH Arrhythmia Database which contains contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979."""

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
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, concatenate, UpSampling1D
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout, Add, LeakyReLU, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

# Load dataset (Specific to ecg dataset)
train_data = pd.read_csv('data/mitbih_train.csv')
test_data = pd.read_csv('data/mitbih_test.csv')

print(train_data.shape)
print(test_data.shape)

train_X = np.array(train_data)[:, :-1]
train_Y = np.array(train_data)[:, -1].astype(int)
test_X = np.array(test_data)[:, :-1]
test_Y = np.array(test_data)[:, -1].astype(int)

# converting the X data to 1 channel data
train_X = train_X.reshape(-1, train_X.shape[1], 1)
test_X = test_X.reshape(-1, test_X.shape[1], 1)

# padding the X data so that it is of length 192 (divisible by 2,4,8,16)
train_X = np.pad(train_X, ((0, 0), (2, 3), (0, 0)), 'constant')
test_X = np.pad(test_X, ((0, 0), (2, 3), (0, 0)), 'constant')
print(train_X.shape)
print(train_Y.shape)


def sr_gen_model(inp_shape, r):  # Assuming r to be a multiple of 2
    inp = Input(shape=inp_shape)
    n = Conv1D(64, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')(inp)
    temp = n

    for i in range(4):  # Number of residual blocks
        nn = Conv1D(64, 3, 1, padding='same', kernel_initializer='he_normal')(n)
        nn = BatchNormalization()(nn)
        nn = Activation('relu')(nn)
        nn = Conv1D(64, 3, 1, padding='same', kernel_initializer='he_normal')(nn)
        nn = BatchNormalization()(nn)
        nn = Add()([n, nn])
        n = nn

    n = Conv1D(64, 3, 1, padding='same', kernel_initializer='he_normal')(n)
    n = BatchNormalization()(n)
    n = Add()([n, temp])

    n_upsample = int(np.log2(r))
    for i in range(n_upsample):
        n = Conv1D(128, 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = UpSampling1D(size=2)(n)
        n = Conv1D(128, 3, 1, padding='same', kernel_initializer='he_normal')(n)

    n = Conv1D(1, 1, 1, padding='same', kernel_initializer='he_normal')(n)
    gen = Model(inputs=inp, outputs=n, name='Generator')
    return gen


sr_gen_model((24, 1), 8).summary()


def sr_disc_model(inp_shape):
    inp = Input(shape=inp_shape)
    outfilters = [32, 64]
    filters = 32
    n = Conv1D(filters, 3, 1, padding='same', kernel_initializer='he_normal')(inp)
    n = LeakyReLU()(n)
    n = Conv1D(filters, 3, 2, padding='same', kernel_initializer='he_normal')(n)
    n = BatchNormalization()(n)
    n = LeakyReLU()(n)

    for i in range(len(outfilters)):
        n = Conv1D(outfilters[i], 3, 1, padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = LeakyReLU()(n)
        n = Conv1D(outfilters[i], 3, 2, padding='same', kernel_initializer='he_normal')(n)
        n = BatchNormalization()(n)
        n = LeakyReLU()(n)

    n = Flatten()(n)
    # n = Dense(256)(n)
    # n = LeakyReLU()(n)
    n = Dense(1)(n)
    n = Activation('sigmoid')(n)

    disc = Model(inputs=inp, outputs=n, name='Discriminator')
    return disc


sr_disc_model((192, 1)).summary()

batch_size = 128
n_epochs_init = 50
n_epochs = 100
init_lr = 1e-4
shuffle_buffer_size = 128
test_batch_size = 128
beta1 = 0.9
save_dir = 'models'


def get_train_data(r):
    train_X_r = train_X[:, ::r, :]

    def train_generator():
        for i in range(len(train_X)):
            yield train_X_r[i], train_X[i]

    train_ds = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float32, tf.float32))
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
    return train_ds


def get_test_data(r):
    test_X_r = test_X[:, ::r, :]

    def test_generator():
        for i in range(len(test_X)):
            yield test_X_r[i], test_X[i]

    test_ds = tf.data.Dataset.from_generator(test_generator, output_types=(tf.float32, tf.float32))
    test_ds = test_ds.prefetch(buffer_size=2)
    test_ds = test_ds.batch(test_batch_size)
    return test_ds


def train_gan(r):
    G = sr_gen_model((train_X.shape[1] // r, 1), r)
    D = sr_disc_model((train_X.shape[1], 1))
    print("Generator")
    print(G.summary())
    print("Discriminator")
    print(D.summary())
    lr_v = tf.Variable(init_lr)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    train_ds = get_train_data(r)
    test_ds = get_test_data(r)
    prev_best = np.inf
    n_steps_train = len(list(train_ds))
    n_steps_test = len(list(test_ds))

    # Initial Learning of Generator to minimize MSE Loss

    # for epoch in range(n_epochs_init):
    #   for step , (lr,hr) in enumerate(train_ds):
    #     if lr.shape[0]<batch_size:
    #       break
    #     step_time = time.time()
    #     with tf.GradientTape(persistent=True) as tape:
    #       hr_f = G(lr,training=True)
    #       loss_g_mse = MeanSquaredError()(hr,hr_f)
    #       loss_g = loss_g_mse

    #     grad = tape.gradient(loss_g,G.trainable_weights)
    #     g_optimizer.apply_gradients(zip(grad, G.trainable_weights))

    #     print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.3f}".format(
    #         epoch, n_epochs_init, step, n_steps_train, time.time() - step_time, loss_g_mse))

    #   test_mse = 0
    #   count = 0
    #   for step,(lr,hr) in enumerate(test_ds):
    #     step_time = time.time()
    #     hr_f = G(lr,training=False)
    #     mse =  MeanSquaredError()(hr_f,hr)
    #     test_mse += mse*lr.shape[0]
    #     count += lr.shape[0]
    #     print("Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.3f}".format(
    #         epoch, n_epochs_init, step, n_steps_test, time.time() - step_time, mse))
    #   test_mse/=count
    #   print("Epoch: [{}/{}] test_mse:{:.3f}".format(epoch,n_epochs_init,test_mse))
    #   if test_mse < prev_best:
    #     G.save(os.path.join(save_dir,'best_gen_sr_'+str(r)+'.pt'))
    #     print('Saving Best generator with best MSE:', test_mse)
    #     prev_best = test_mse
    #   G.save(os.path.join(save_dir,'last_gen_model_sr_'+str(r)+'.pt'))

    # Training Discriminator Alongside Generator
    for epoch in range(n_epochs):
        for step, (lr, hr) in enumerate(train_ds):
            if lr.shape[0] < batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                hr_f = G(lr, training=True)
                logits_f = D(hr_f, training=True)
                logits_r = D(hr, training=True)
                loss_d_1 = MeanSquaredError()(logits_f, tf.zeros_like(logits_f))
                loss_d_2 = MeanSquaredError()(logits_r, 0.9 * tf.ones_like(logits_r))  # Label Smoothing
                loss_d = loss_d_1 + loss_d_2
                loss_g_gan = MeanSquaredError()(logits_f, 0.9 * tf.ones_like(logits_f))  # Label Smoothing
                loss_g_mse = MeanSquaredError()(hr, hr_f)
                loss_g = loss_g_mse + 1e-3 * loss_g_gan

            grad = tape.gradient(loss_d, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            grad = tape.gradient(loss_g, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))

            print(
                "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.3f},  adv:{:.3f},  d_fake_loss: {:.7f}, d_real_loss: {:.7f} ".format(
                    epoch, n_epochs, step, n_steps_train, time.time() - step_time, loss_g_mse, loss_g_gan, loss_d_1,
                    loss_d_2))

        # G.eval()
        test_mse = 0
        count = 0
        for step, (lr, hr) in enumerate(test_ds):
            step_time = time.time()
            hr_f = G(lr, training=False)
            mse = MeanSquaredError()(hr_f, hr)
            test_mse += mse * lr.shape[0]
            count += lr.shape[0]
            print("Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.3f}".format(
                epoch, n_epochs, step, n_steps_test, time.time() - step_time, mse))
        test_mse /= count
        print("Epoch: [{}/{}] test_mse:{:.3f}".format(epoch, n_epochs, test_mse))
        if test_mse < prev_best:
            G.save(os.path.join(save_dir, 'best_gen_sr_' + str(r) + '.pt'))
            print('Saving Best generator with best MSE:', test_mse)
            prev_best = test_mse
        G.save(os.path.join(save_dir, 'last_gen_model_sr_' + str(r) + '.pt'))
        D.save(os.path.join(save_dir, 'last_disc_model_sr_' + str(r) + '.pt'))


if __name__=='__main__':
    sampling_rates = [8]
    for r in sampling_rates:
        print("Training GAN for sampling rate ", r)
        train_gan(r)
