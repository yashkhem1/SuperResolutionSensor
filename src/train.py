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


def gradient_penalty(f, real, fake):
    '''
    Gradient penalty for WGAN-GP
    :param f: Keras Model
    :param real: Tensorflow Tensor
    :param fake: Tensorflow Tensor
    :return: Tensorflow Scalar Tensor
    '''
    def _interpolate(a, b):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter

    x = _interpolate(real, fake)
    with tf.GradientTape() as t:
        t.watch(x)
        pred = f(x)
    grad = t.gradient(pred, x)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.)**2)
    return gp



def train_clf(opt):
    '''
    Training loop for classification
    :param opt: Argument Parser
    :return:
    '''
    if opt.data_type == 'ecg':
        inp_shape = (192//opt.sampling_ratio,1)
        if opt.use_sr_clf:
            inp_shape = (192,1)
        nclasses =  5
        C = clf_model_func('ecg')(inp_shape,nclasses)

    print(C.summary())
    lr_v = tf.Variable(opt.init_lr)
    c_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)
    sr_model = None
    if opt.use_sr_clf:
        sr_model = opt.model_path

    train_ds = train_cf_dataset(opt.data_type,opt.sampling_ratio,opt.train_batch_size,opt.shuffle_buffer_size,opt.fetch_buffer_size,
                                opt.resample,sr_model)

    test_ds = test_cf_dataset(opt.data_type,opt.sampling_ratio,opt.test_batch_size,opt.fetch_buffer_size,sr_model)
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
        f1_train = f1_score(y_true_train,y_pred_train,average='macro')
        print("Epoch: [{}/{}]  accuracy:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, accuracy_train, f1_train))

        #Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.init_lr * new_lr_decay)
            print("New learning rate", opt.init_lr * new_lr_decay)

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
        f1_test = f1_score(y_true_test,y_pred_test,average='macro')
        print("Epoch: [{}/{}] test_accuracy:{:.6f}, test_f1_score:{:.6f}".format(epoch, opt.epochs, accuracy_test, f1_test))
        if opt.use_sr_clf:
            model_defs = opt.model_path.split('/')[-1].split('_')
            sr_string = model_defs[1]
            use_perception = model_defs[4]
        else:
            sr_string = '0'
            use_perception='0'
        if accuracy_test > prev_best:
            C.save(os.path.join(opt.save_dir, 'best_clf_' + str(opt.data_type) + '_sampling_'+str(opt.sampling_ratio)
                                + '_sr_' + sr_string + '_perception_'+ use_perception+'_resample_'+ str(opt.resample) +
                                '_weighted_' + str(opt.weighted) + '.pt'))
            print('Saving Best generator with best accuracy:', accuracy_test, 'and F1 score:', f1_test)
            prev_best = accuracy_test
        C.save(os.path.join(opt.save_dir, 'last_clf_' + str(opt.data_type) + '_sampling_'+str(opt.sampling_ratio)
                                + '_sr_' + sr_string + '_perception_'+ use_perception+'_resample_'+ str(opt.resample) +
                                '_weighted_' + str(opt.weighted) + '.pt'))


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
        x_true_train = []
        x_pred_train = []
        y_true_train = np.array([],dtype='int32')
        y_pred_train = np.array([],dtype='int32')
        for step , (lr,hr,y) in enumerate(train_ds):
            if lr.shape[0]<opt.train_batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                hr_f = G(lr,training=True)
                loss_mse = MeanSquaredError()(hr,hr_f)
                loss_pr = 0
                if opt.use_perception_loss:
                    p_f = P(hr_f,training=False)
                    p_r = P(hr,training=False)
                    loss_pr = MeanSquaredError()(p_r,p_f)
                loss_g = loss_mse + 1e-3*loss_pr

            grad = tape.gradient(loss_g,G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            x_true_train +=list(hr.numpy())
            x_pred_train +=list(hr_f.numpy())
            y_true = np.argmax(C(hr,training=False),axis=1)
            y_pred = np.argmax(C(hr_f,training=False),axis=1)
            y_true_train = np.append(y_true_train, y_true)
            y_pred_train = np.append(y_pred_train, y_pred)
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.6f}, perception_loss:{:.6f}".format(epoch,
                    opt.epochs,step, n_steps_train,time.time() - step_time, loss_mse,loss_pr))

        x_true_train = np.array(x_true_train)
        x_pred_train = np.array(x_pred_train)
        train_mse = np.mean((x_true_train-x_pred_train)**2)
        train_task_score = accuracy_score(y_true_train,y_pred_train)
        print("Epoch: [{}/{}]  mse:{:.6f}, accuracy_score:{:.6f} ".format(
            epoch, opt.epochs, train_mse, train_task_score))

        # Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.init_lr * new_lr_decay)
            print("New learning rate", opt.init_lr * new_lr_decay)

        x_true_test = []
        x_pred_test = []
        y_true_test = np.array([], dtype='int32')
        y_pred_test = np.array([], dtype='int32')
        for step,(lr,hr,y) in enumerate(test_ds):
            step_time = time.time()
            hr_f = G(lr,training=False)
            x_true_test += list(hr.numpy())
            x_pred_test += list(hr_f.numpy())
            y_true = np.argmax(C(hr, training=False), axis=1)
            y_pred = np.argmax(C(hr_f, training=False), axis=1)
            y_true_test = np.append(y_true_test, y_true)
            y_pred_test = np.append(y_pred_test, y_pred)
            print("Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time))

        x_true_test = np.array(x_true_test)
        x_pred_test = np.array(x_pred_test)
        test_mse = np.mean((x_true_test- x_pred_test)**2)
        test_task_score= accuracy_score(y_true_test, y_pred_test)
        print("Epoch: [{}/{}]  mse:{:.6f}, accuracy_score:{:.6f} ".format(
            epoch, opt.epochs, test_mse, test_task_score))
        if test_mse < prev_best:
            G.save(os.path.join(opt.save_dir,'best_cnn_'+str(opt.data_type)+'_'+str(opt.sampling_ratio)+'_'+str(opt.use_perception_loss)+'.pt'))
            print('Saving Best generator with best MSE:', test_mse)
            prev_best = test_mse
        G.save(os.path.join(opt.save_dir,'last_cnn_'+str(opt.data_type)+'_'+str(opt.sampling_ratio)+'_'+str(opt.use_perception_loss)+'.pt'))


def train_sr_gan(opt):
    '''
    Training loop for super resolution with adversarial loss
    :param opt: Argument Parser
    :return:
    '''
    if opt.data_type == 'ecg':
        inp_shape = (192//opt.sampling_ratio,1)
        inp_disc_shape = (192,1)
        nclasses =  5
        G = sr_model_func('ecg')(inp_shape,opt.sampling_ratio)
        D = disc_model_func('ecg')(inp_disc_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)


    print(G.summary())
    print(D.summary())
    lr_v = tf.Variable(opt.init_lr)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)

    train_ds = train_sr_dataset(opt.data_type, opt.sampling_ratio, opt.train_batch_size, opt.shuffle_buffer_size,
                                opt.fetch_buffer_size, opt.resample)
    test_ds = test_sr_dataset(opt.data_type, opt.sampling_ratio, opt.test_batch_size, opt.fetch_buffer_size)
    prev_best = np.inf
    n_steps_train = len(list(train_ds))
    n_steps_test = len(list(test_ds))

    for epoch in range(opt.epochs):
        x_true_train = []
        x_pred_train = []
        y_true_train = np.array([],dtype='int32')
        y_pred_train = np.array([],dtype='int32')
        for step , (lr,hr,y) in enumerate(train_ds):
            if lr.shape[0]<opt.train_batch_size:
                break
            step_time = time.time()
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                hr_f = G(lr,training=True)
                loss_mse = MeanSquaredError()(hr,hr_f)
                loss_gen = 0
                f_loss= 0
                r_loss= 0
                loss_d=0
                loss_pr = 0
                grad_p = 0
                if epoch>=opt.init_epochs:
                    logits_f = D(hr_f, training=True)
                    logits_r = D(hr, training=True)
                    if opt.gan_type == 'normal':
                        f_loss = MeanSquaredError()(tf.zeros_like(logits_f),logits_f)
                        r_loss = MeanSquaredError()(tf.ones_like(logits_r),logits_r)
                        loss_d = f_loss + r_loss
                        loss_gen = MeanSquaredError()(tf.ones_like(logits_f),logits_f)
                    elif opt.gan_type == 'normal_ls':
                        f_loss = MeanSquaredError()(tf.zeros_like(logits_f) + tf.random.uniform(logits_f.shape,0,1)*0.3,logits_f) # Label Smoothing
                        r_loss = MeanSquaredError()(tf.ones_like(logits_r) - 0.3 + tf.random.uniform(logits_f.shape,0,1)*0.5,logits_r)  # Label Smoothing
                        loss_d = f_loss + r_loss
                        loss_gen = MeanSquaredError()(tf.ones_like(logits_f) - 0.3 + tf.random.uniform(logits_f.shape,0,1)*0.5,logits_f)  # Label Smoothing
                    elif opt.gan_type == 'wgan':
                        r_loss = - tf.reduce_mean(logits_r)
                        f_loss = tf.reduce_mean(logits_f)
                        loss_d = f_loss+r_loss
                        loss_gen = -tf.reduce_mean(logits_f)
                    elif opt.gan_type =='wgan_gp':
                        r_loss = - tf.reduce_mean(logits_r)
                        f_loss = tf.reduce_mean(logits_f)
                        grad_p = gradient_penalty(D,hr,hr_f)
                        loss_d = f_loss + r_loss + opt.gp_lambda*grad_p
                        loss_gen = -tf.reduce_mean(logits_f)

                if opt.use_perception_loss:
                    p_f = P(hr_f,training=False)
                    p_r = P(hr,training=False)
                    loss_pr = MeanSquaredError()(p_r,p_f)

                loss_g = loss_mse + 1e-3*loss_pr + 1e-3*loss_gen


            grad = gen_tape.gradient(loss_g,G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            if epoch>=opt.init_epochs:
                grad_d = disc_tape.gradient(loss_d,D.trainable_weights)
                d_optimizer.apply_gradients(zip(grad_d,D.trainable_weights))
                if opt.gan_type =='WGAN':
                    for l in D.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -opt.clip_value, opt.clip_value) for w in weights]
                        l.set_weights(weights)

            x_true_train += list(hr.numpy())
            x_pred_train += list(hr_f.numpy())
            y_true = np.argmax(y,axis=1)
            y_pred = np.argmax(C(hr_f,training=False),axis=1)
            y_true_train = np.append(y_true_train, y_true)
            y_pred_train = np.append(y_pred_train, y_pred)
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.6f}, perception_loss:{:.6f}, adv:{:.6f}, d_fake_loss:{:.6f},"
                  " d_real_loss:{:.6f},  gradient_penalty:{:.6f}".format(epoch,
                    opt.epochs,step, n_steps_train,time.time() - step_time, loss_mse,loss_pr,loss_gen,f_loss,r_loss,grad_p))

        x_true_train = np.array(x_true_train)
        x_pred_train = np.array(x_pred_train)
        train_mse = np.mean((x_true_train-x_pred_train)**2)
        train_task_score = accuracy_score(y_true_train,y_pred_train)
        print("Epoch: [{}/{}]  mse:{:.6f}, accuracy_score:{:.6f} ".format(
            epoch, opt.epochs, train_mse, train_task_score))

        # Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.init_lr * new_lr_decay)
            print("New learning rate", opt.init_lr * new_lr_decay)

        x_true_test = []
        x_pred_test = []
        y_true_test = np.array([], dtype='int32')
        y_pred_test = np.array([], dtype='int32')
        for step,(lr,hr,y) in enumerate(test_ds):
            step_time = time.time()
            hr_f = G(lr,training=False)
            x_true_test += list(hr.numpy())
            x_pred_test += list(hr_f.numpy())
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(C(hr_f, training=False), axis=1)
            y_true_test = np.append(y_true_test, y_true)
            y_pred_test = np.append(y_pred_test, y_pred)
            print("Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time))

        x_true_test = np.array(x_true_test)
        x_pred_test = np.array(x_pred_test)
        test_mse = np.mean((x_true_test - x_pred_test)**2)
        test_task_loss = accuracy_score(y_true_test, y_pred_test)
        print("Epoch: [{}/{}]  mse:{:.6f}, accuracy_score:{:.6f} ".format(
            epoch, opt.epochs, test_mse, test_task_loss))
        if test_mse < prev_best:
            G.save(os.path.join(opt.save_dir,'best_gen_'+str(opt.gan_type)+'_'+str(opt.data_type)+'_'+str(opt.sampling_ratio)+'_'+str(opt.use_perception_loss)+'.pt'))
            D.save(os.path.join(opt.save_dir,
                                'best_disc_' + str(opt.data_type) + '_' + str(opt.sampling_ratio) + '_' + str(
                                    opt.use_perception_loss) + '.pt'))
            print('Saving Best generator with best MSE:', test_mse)
            prev_best = test_mse
        G.save(os.path.join(opt.save_dir,'last_gen_'+str(opt.gan_type)+'_'+str(opt.data_type)+'_'+str(opt.sampling_ratio)+'_'+str(opt.use_perception_loss)+'.pt'))
        D.save(os.path.join(opt.save_dir, 'last_disc_'+str(opt.gan_type)+'_' + str(opt.data_type) + '_' + str(opt.sampling_ratio) + '_' + str(
            opt.use_perception_loss) + '.pt'))










