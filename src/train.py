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
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy,CategoricalCrossentropy
from src.models import *
from src.data_loader import *

def train_clf(opt):
    '''
    Training loop for classification
    :param opt: Argument Parser
    :return:
    '''
    if opt.data_type == 'ecg':
        inp_shape = (192,1)
        nclasses =  5
        C = clf_model_func('ecg')(inp_shape,nclasses)

    lr_v = tf.Variable(opt.init_lr)
    c_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)

    train_ds = train_cf_dataset(opt.data_type,1,opt.train_batch_size,opt.shuffle_buffer_size,opt.fetch_buffer_size,
                                opt.resample)

    test_ds = test_cf_dataset(opt.data_type,1,opt.test_batch_size,opt.fetch_buffer_size)
    prev_best = -1*np.inf
    n_steps_train = len(list(train_ds))
    n_steps_test = len(list(test_ds))


    for epoch in range(opt.epochs):
        y_true_train = []
        y_pred_train = []
        for step, (X, y_true) in enumerate(train_ds):
            if X.shape[0] < opt.train_batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                y_pred = C(X)
                if opt.weighted:
                    class_weights = get_class_weights(opt.data_type)
                else:
                    class_weights = np.array(1.0/nclasses)

                weights = tf.reduce_sum(class_weights*y_true,axis=1)
                unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred)
                loss = weights*unweighted_loss
                loss = tf.reduce_mean(loss)

            grad = tape.gradient(loss, C.trainable_weights)
            c_optimizer.apply_gradients(zip(grad, C.trainable_weights))
            y_true_train.append(np.argmax(y_true,axis=1))
            y_pred_train.append(np.argmax(y_pred,axis=1))

            print(
                "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, cross_entropy_loss:{:.6f} ".format(
                    epoch, opt.epochs, step, n_steps_train, time.time() - step_time, loss))

        accuracy_train = accuracy_score(y_true_train,y_pred_train)
        f1_train = f1_score(y_true_train,y_pred_train)
        print("Epoch: [{}/{}]  accuracy:{:.6f}, f1_score:{:.6f} ".format(
                    epoch, opt.epochs, accuracy_train, f1_train))

        y_true_test = []
        y_pred_test = []
        for step, (X, y_true) in enumerate(test_ds):
            step_time = time.time()
            y_pred = C(X, training=False)
            y_true_test.append(np.argmax(y_true,axis=1))
            y_pred_test.append(np.argmax(y_pred,axis=1))
            print("Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time))

        accuracy_test = accuracy_score(y_true_test,y_pred_test)
        f1_test = f1_score(y_true_test,y_pred_test)
        print("Epoch: [{}/{}] test_accuracy:{:.6f}, test_f1_score:{:.6f}".format(epoch, opt.epochs, accuracy_test, f1_test))
        if f1_test > prev_best:
            C.save(os.path.join(opt.save_dir, 'best_clf_' + str(opt.data_type) + '.pt'))
            print('Saving Best generator with best accuracy:', accuracy_test, 'and F1 score:', f1_test)
            prev_best = f1_test
        C.save(os.path.join(opt.save_dir, 'last_clf_' + str(opt.data_type) + '.pt'))







