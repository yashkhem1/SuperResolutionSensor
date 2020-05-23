import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,mean_squared_error
from sklearn.utils import class_weight

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, concatenate, UpSampling1D
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout, Add, LeakyReLU, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy,CategoricalCrossentropy
from tensorflow.keras.models import load_model
from src.models import *
from src.data_loader import *


def train_clf(opt):
    '''
    Training loop for classification
    :param opt: Argument Parser
    :return:
    '''
    if opt.data_type == 'ecg':
        inp_shape = (192//opt.sampling_ratio,1)
        nclasses =  5
        C = clf_model_func('ecg')(inp_shape,nclasses)

    print(C.summary())
    lr_v = tf.Variable(opt.init_lr)
    c_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)

    train_ds = train_cf_dataset(opt.data_type,opt.sampling_ratio,opt.train_batch_size,opt.shuffle_buffer_size,opt.fetch_buffer_size,
                                opt.resample)

    test_ds = test_cf_dataset(opt.data_type,opt.sampling_ratio,opt.test_batch_size,opt.fetch_buffer_size)
    prev_best = -1*np.inf
    n_steps_train = len(list(train_ds))
    n_steps_test = len(list(test_ds))

    if opt.weighted:
        class_weights = get_class_weights(opt.data_type)
    else:
        class_weights = np.array([1.0 / nclasses] * nclasses)

    for epoch in range(opt.epochs):
        y_true_train = np.array([],dtype='int32')
        y_pred_train = np.array([],dtype='int32')
        for step, (X, y_true) in enumerate(train_ds):
            if X.shape[0] < opt.train_batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                y_pred = C(X,training=True)
                # print(class_weights)
                # print(y_true)
                weights = tf.reduce_sum(class_weights*y_true,axis=1)
                unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred)
                # print(weights)
                # print(unweighted_loss)
                loss = weights*unweighted_loss
                loss = tf.reduce_mean(loss)

            grad = tape.gradient(loss, C.trainable_weights)
            c_optimizer.apply_gradients(zip(grad, C.trainable_weights))
            y_true_train = np.append(y_true_train,np.argmax(y_true,axis=1))
            y_pred_train = np.append(y_pred_train,np.argmax(y_pred,axis=1))

            print(
                "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, cross_entropy_loss:{:.6f} ".format(
                    epoch, opt.epochs, step, n_steps_train, time.time() - step_time, loss))

        # print(y_true_train)
        # print(y_pred_train)
        accuracy_train = accuracy_score(y_true_train,y_pred_train)
        f1_train = f1_score(y_true_train,y_pred_train,average='weighted')
        print("Epoch: [{}/{}]  accuracy:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, accuracy_train, f1_train))

        #Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.lr_init * new_lr_decay)
            print("New learning rate", opt.lr_init * new_lr_decay)

        y_true_test = np.array([],dtype='int32')
        y_pred_test = np.array([],dtype='int32')
        for step, (X, y_true) in enumerate(test_ds):
            step_time = time.time()
            y_pred = C(X, training=False)
            y_true_test = np.append(y_true_test,np.argmax(y_true,axis=1))
            y_pred_test = np.append(y_pred_test,np.argmax(y_pred,axis=1))
            print("Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time))

        accuracy_test = accuracy_score(y_true_test,y_pred_test)
        f1_test = f1_score(y_true_test,y_pred_test,average='weighted')
        print("Epoch: [{}/{}] test_accuracy:{:.6f}, test_f1_score:{:.6f}".format(epoch, opt.epochs, accuracy_test, f1_test))
        if accuracy_test > prev_best:
            C.save(os.path.join(opt.save_dir, 'best_clf_' + str(opt.data_type) + '.pt'))
            print('Saving Best generator with best accuracy:', accuracy_test, 'and F1 score:', f1_test)
            prev_best = accuracy_test
        C.save(os.path.join(opt.save_dir, 'last_clf_' + str(opt.data_type) + '.pt'))


def train_sr(opt):
    '''
    Training loop for super resolution without adversarial loss
    :param opt: Argument Parser
    :return:
    '''
    if opt.data_type == 'ecg':
        inp_shape = (192//opt.sampling_ratio,1)
        nclasses =  5
        G = sr_model_func('ecg')(inp_shape,opt.sampling_ratio)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)


    print(G.summary())
    lr_v = tf.Variable(opt.init_lr)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)

    train_ds = train_sr_dataset(opt.data_type, opt.sampling_ratio, opt.train_batch_size, opt.shuffle_buffer_size,
                                opt.fetch_buffer_size, opt.resample)
    test_ds = test_sr_dataset(opt.data_type, opt.sampling_ratio, opt.test_batch_size, opt.fetch_buffer_size)
    prev_best = np.inf
    n_steps_train = len(list(train_ds))
    n_steps_test = len(list(test_ds))

    for epoch in range(opt.epochs):
        x_true_train = np.array([])
        x_pred_train = np.array([])
        y_true_train = np.array([],dtype='int32')
        y_pred_train = np.array([],dtype='int32')
        for step , (lr,hr,y) in enumerate(train_ds):
            if lr.shape[0]<opt.train_batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                hr_f = G(lr,training=False)
                loss_mse = MeanSquaredError()(hr,hr_f)
                loss_pr = 0
                if opt.use_perception_loss:
                    p_f = P(hr_f,training=False)
                    p_r = P(hr,training=False)
                    loss_pr = MeanSquaredError()(p_r,p_f)
                loss_g = loss_mse + loss_pr

            grad = tape.gradient(loss_g,G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            x_true_train = np.append(x_true_train,hr)
            x_pred_train = np.append(x_pred_train,hr_f)
            y_true = np.argmax(C(hr,training=False),axis=1)
            y_pred = np.argmax(C(hr_f,training=False),axis=1)
            y_true_train = np.append(y_true_train, y_true)
            y_pred_train = np.append(y_pred_train, y_pred)
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.6f}, perception_loss:{:.6f}".format(epoch,
                    opt.epochs,step, n_steps_train,time.time() - step_time, loss_mse,loss_pr))

        train_mse = mean_squared_error(x_true_train,x_pred_train)
        train_task_loss = accuracy_score(y_true_train,y_pred_train)
        print("Epoch: [{}/{}]  mse:{:.6f}, task_loss:{:.6f} ".format(
            epoch, opt.epochs, train_mse, train_task_loss))

        x_true_test = np.array([])
        x_pred_test = np.array([])
        y_true_test = np.array([], dtype='int32')
        y_pred_test = np.array([], dtype='int32')
        for step,(lr,hr,y) in enumerate(test_ds):
            step_time = time.time()
            hr_f = G(lr,training=False)
            x_true_test = np.append(x_true_test, hr)
            x_pred_test = np.append(x_pred_test, hr_f)
            y_true = np.argmax(C(hr, training=False), axis=1)
            y_pred = np.argmax(C(hr_f, training=False), axis=1)
            y_true_test = np.append(y_true_test, y_true)
            y_pred_test = np.append(y_pred_test, y_pred)
            print("Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time))

        test_mse = mean_squared_error(x_true_test, x_pred_test)
        test_task_loss = accuracy_score(y_true_test, y_pred_test)
        print("Epoch: [{}/{}]  mse:{:.6f}, task_loss:{:.6f} ".format(
            epoch, opt.epochs, test_mse, test_task_loss))
        if test_mse < prev_best:
            G.save(os.path.join(opt.save_dir,'best_gen_'+str(opt.data_type)+'.pt'))
            print('Saving Best generator with best MSE:', test_mse)
            prev_best = test_mse
        G.save(os.path.join(opt.save_dir,'last_gen_'+str(opt.data_type)+'.pt'))










