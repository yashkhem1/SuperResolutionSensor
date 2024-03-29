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

from progress.bar import Bar
import sys


def gradient_penalty(f, real, fake):
    '''
    Gradient penalty for WGAN-GP
    :param f: Keras Model
    :param real: Tensorflow Tensor
    :param fake: Tensorflow Tensor
    :return: Tenfsorflow Scalar Tensor
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



# ===============================================================
#                     Classification
# ===============================================================


def train_clf(opt):
    '''
    Training loop for classification
    :param opt: Argument Parser
    :return:
    '''
    if opt.data_type == 'ecg':
        inp_shape = (192//opt.sampling_ratio,1)
        if opt.use_sr_clf or opt.interp:
            inp_shape = (192,1)
        nclasses =  5
        C = clf_model_func('ecg')(inp_shape,nclasses)

    elif opt.data_type == 'shl':
        inp_shape = (6,512//opt.sampling_ratio,1)
        if opt.use_sr_clf or opt.interp:
            inp_shape = (6,512,1)
        nclasses = 8
        C = clf_model_func('shl')(inp_shape,nclasses)

    elif opt.data_type == 'audio':
        inp_shape = (8000//opt.sampling_ratio,1)
        if opt.use_sr_clf or opt.interp:
            inp_shape = (8000,1)
        nclasses =  10
        C = clf_model_func('audio')(inp_shape,nclasses)

    elif opt.data_type == 'pam2':
        inp_shape = (27,256//opt.sampling_ratio,1)
        if opt.use_sr_clf or opt.interp:
            inp_shape = (27,256,1)
        nclasses = 12
        C = clf_model_func('pam2')(inp_shape,nclasses)

    print(C.summary())
    lr_v = tf.Variable(opt.init_lr)
    c_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)
    sr_model = None
    imp_model  = None
    if opt.use_sr_clf:
        sr_model = opt.model_path
    if opt.use_imp_clf:
        imp_model = opt.model_path

    train_ds = train_cf_dataset(opt.data_type,opt.sampling_ratio,opt.train_batch_size,opt.shuffle_buffer_size,opt.fetch_buffer_size,
                                opt.resample,sr_model,opt.interp, opt.interp_type, opt.prob,opt.seed,opt.cont,opt.fixed,imp_model)

    test_ds = test_cf_dataset(opt.data_type,opt.sampling_ratio,opt.test_batch_size,opt.fetch_buffer_size,sr_model,opt.interp,
                              opt.interp_type,opt.prob, opt.seed, opt.cont, imp_model)

    prev_best = -1*np.inf
    n_steps_train = len(list(train_ds))
    n_steps_test = len(list(test_ds))

    if opt.weighted:
        class_weights = get_class_weights(opt.data_type)*nclasses
    else:
        class_weights = np.array([1.0] * nclasses)

    for epoch in range(opt.epochs):
        y_true_train = np.array([],dtype='int32')
        y_pred_train = np.array([],dtype='int32')
        bar = Bar('>>>', fill='>', max=n_steps_train)
        epoch_time = time.time()
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
                # loss = CategoricalCrossentropy(from_logits=True)(y_true,y_pred)

            grad = tape.gradient(loss, C.trainable_weights)
            c_optimizer.apply_gradients(zip(grad, C.trainable_weights))
            y_true_train = np.append(y_true_train,np.argmax(y_true,axis=1))
            y_pred_train = np.append(y_pred_train,np.argmax(y_pred,axis=1))

            bar.suffix = "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, ce_loss:{:.6f} ".format(
                    epoch, opt.epochs, step, n_steps_train, time.time() - step_time, loss)
            bar.next()

        # print(y_true_train)
        # print(y_pred_train)
        bar.finish()
        accuracy_train = accuracy_score(y_true_train,y_pred_train)
        f1_train = f1_score(y_true_train,y_pred_train,average='macro')
        print("Epoch: [{}/{}] time:{:.1f}s, accuracy:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, time.time()-epoch_time, accuracy_train, f1_train))

        #Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.init_lr * new_lr_decay)
            print("New learning rate " + str(opt.init_lr * new_lr_decay))

        y_true_test = np.array([],dtype='int32')
        y_pred_test = np.array([],dtype='int32')
        bar = Bar('>>>', fill='>', max=n_steps_test)
        epoch_time = time.time()
        for step, (X, y_true) in enumerate(test_ds):
            step_time = time.time()
            y_pred = C(X, training=False)
            y_true_test = np.append(y_true_test,np.argmax(y_true,axis=1))
            y_pred_test = np.append(y_pred_test,np.argmax(y_pred,axis=1))
            bar.suffix  = "Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time)
            bar.next()
        
        bar.finish()

        accuracy_test = accuracy_score(y_true_test,y_pred_test)
        f1_test = f1_score(y_true_test,y_pred_test,average='macro')
        print("Testing Epoch: [{}/{}] time:{:.1f}s, test_accuracy:{:.6f}, test_f1_score:{:.6f}".format(epoch, opt.epochs, time.time()-epoch_time, accuracy_test, f1_test))
        if opt.use_sr_clf:
            model_defs = opt.model_path.split('/')[-1].split('_')
            sr_string = model_defs[1]
            use_perception = model_defs[4][:-5]

        elif opt.interp:
            sr_string = opt.interp_type
            use_perception = '0'

        else:
            sr_string = '0'
            use_perception='0'

        clf_string = 'clf'

        if opt.prob!=0:
            if opt.use_imp_clf:
                model_defs = opt.model_path.split('/')[-1].split('_')
                imp_string = model_defs[1]
                use_perception = model_defs[4]
                masked_loss = model_defs[5][:-5]
            else:
                imp_string = '0'
                use_perception = '0'
                masked_loss = '0'

            if opt.fixed:
                clf_string = 'clf-fixed'


        if f1_test > prev_best:
            if opt.prob==0:
                C.save(os.path.join(opt.save_dir, 'best_'+clf_string+'_' + str(opt.data_type) + '_sampling_'+str(opt.sampling_ratio)
                                    + '_sr_' + sr_string + '_perception_'+ use_perception+'_resample_'+ str(opt.resample) +
                                    '_weighted_' + str(opt.weighted) + '.hdf5'))
            else:
                C.save(os.path.join(opt.save_dir, 'best_'+clf_string+'_' + str(opt.data_type) + '_prob_' + str(opt.prob)
                                 + '_imp_' + imp_string + '_perception_' + use_perception + '_maskedloss_'+masked_loss+
                                '_resample_' + str(opt.resample) + '_weighted_' + str(opt.weighted) + '.hdf5'))
            print('Saving Best generator with best accuracy:' + str(accuracy_test)+ ' and F1 score:' + str(f1_test))
            prev_best = f1_test
        if opt.prob==0:
            C.save(os.path.join(opt.save_dir, 'last_'+clf_string+'_'+ str(opt.data_type) + '_sampling_'+str(opt.sampling_ratio)
                                    + '_sr_' + sr_string + '_perception_'+ use_perception+'_resample_'+ str(opt.resample) +
                                    '_weighted_' + str(opt.weighted) + '.hdf5'))
        else:
            C.save(os.path.join(opt.save_dir, 'last_'+clf_string+'_' + str(opt.data_type) + '_prob_' + str(opt.prob)
                                + '_imp_' + imp_string + '_perception_' + use_perception + '_maskedloss_' + masked_loss +
                                '_resample_' + str(opt.resample) + '_weighted_' + str(opt.weighted) + '.hdf5'))



# ===============================================================
#                     Super Resolution with CNN
# ===============================================================


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

    elif opt.data_type == 'shl':
        inp_shape = (6,512//opt.sampling_ratio,1)
        nclasses =  8
        G = sr_model_func('shl')(inp_shape,opt.sampling_ratio)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'audio':
        inp_shape = (8000//opt.sampling_ratio,1)
        nclasses =  10
        G = sr_model_func('audio')(inp_shape,opt.sampling_ratio)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'pam2':
        inp_shape = (27,256//opt.sampling_ratio,1)
        nclasses =  12
        G = sr_model_func('pam2')(inp_shape,opt.sampling_ratio)
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
        bar = Bar('>>>', fill='>', max=n_steps_train)
        epoch_time = time.time()
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
                loss_g = loss_mse + opt.pl_lambda*loss_pr

            grad = tape.gradient(loss_g,G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            x_true_train +=list(hr.numpy())
            x_pred_train +=list(hr_f.numpy())
            y_true = np.argmax(y,axis=1)
            y_pred = np.argmax(C(hr_f,training=False),axis=1)
            y_true_train = np.append(y_true_train, y_true)
            y_pred_train = np.append(y_pred_train, y_pred)
            bar.suffix = ("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.6f}, p_loss:{:.6f}".format(epoch,
                    opt.epochs,step, n_steps_train,time.time() - step_time, loss_mse,loss_pr))
            bar.next()

        bar.finish()
        x_true_train = np.array(x_true_train)
        x_pred_train = np.array(x_pred_train)
        train_mse = np.mean((x_true_train-x_pred_train)**2)
        train_task_score = f1_score(y_true_train,y_pred_train,average='macro')
        print("Epoch: [{}/{}] time:{:.1f}s, mse:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, time.time()-epoch_time, train_mse, train_task_score))

        # Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.init_lr * new_lr_decay)
            print("New learning rate " + str(opt.init_lr * new_lr_decay))

        x_true_test = []
        x_pred_test = []
        y_true_test = np.array([], dtype='int32')
        y_pred_test = np.array([], dtype='int32')
        bar = Bar('>>>', fill='>', max=n_steps_test)
        epoch_time = time.time()
        for step,(lr,hr,y) in enumerate(test_ds):
            step_time = time.time()
            hr_f = G(lr,training=False)
            x_true_test += list(hr.numpy())
            x_pred_test += list(hr_f.numpy())
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(C(hr_f, training=False), axis=1)
            y_true_test = np.append(y_true_test, y_true)
            y_pred_test = np.append(y_pred_test, y_pred)
            bar.suffix = ("Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time))
            bar.next()
        
        bar.finish()
        x_true_test = np.array(x_true_test)
        x_pred_test = np.array(x_pred_test)
        test_mse = np.mean((x_true_test- x_pred_test)**2)
        test_task_score= f1_score(y_true_test, y_pred_test,average='macro')
        print("Testing Epoch: [{}/{}] time:{:.1f}s, mse:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, time.time()-epoch_time,test_mse, test_task_score))
        if test_mse < prev_best:
            G.save(os.path.join(opt.save_dir,'best_cnn_'+str(opt.data_type)+'_'+str(opt.sampling_ratio)+'_'+str(opt.use_perception_loss)+'.hdf5'))
            print('Saving Best classifier with best MSE:'+ str(test_mse))
            prev_best = test_mse
        G.save(os.path.join(opt.save_dir,'last_cnn_'+str(opt.data_type)+'_'+str(opt.sampling_ratio)+'_'+str(opt.use_perception_loss)+'.hdf5'))


# ===============================================================
#                     Super Resolution with GAN
# ===============================================================


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

    elif opt.data_type == 'shl':
        inp_shape = (6,512//opt.sampling_ratio,1)
        inp_disc_shape = (6,512,1)
        nclasses =  8
        G = sr_model_func('shl')(inp_shape,opt.sampling_ratio)
        D = disc_model_func('shl')(inp_disc_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'audio':
        inp_shape = (8000//opt.sampling_ratio,1)
        inp_disc_shape = (8000,1)
        nclasses =  10
        G = sr_model_func('audio')(inp_shape,opt.sampling_ratio)
        D = disc_model_func('audio')(inp_disc_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'pam2':
        inp_shape = (27,256//opt.sampling_ratio,1)
        inp_disc_shape = (27,256,1)
        nclasses =  12
        G = sr_model_func('pam2')(inp_shape,opt.sampling_ratio)
        D = disc_model_func('pam2')(inp_disc_shape)
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
        bar = Bar('>>>', fill='>', max=n_steps_train)
        epoch_time = time.time()
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
                    elif opt.gan_type == 'normalls':
                        f_loss = MeanSquaredError()(tf.zeros_like(logits_f) + tf.random.uniform(logits_f.shape,0,1)*0.3,logits_f) # Label Smoothing
                        r_loss = MeanSquaredError()(tf.ones_like(logits_r) - 0.3 + tf.random.uniform(logits_f.shape,0,1)*0.5,logits_r)  # Label Smoothing
                        loss_d = f_loss + r_loss
                        loss_gen = MeanSquaredError()(tf.ones_like(logits_f) - 0.3 + tf.random.uniform(logits_f.shape,0,1)*0.5,logits_f)  # Label Smoothing
                    elif opt.gan_type == 'wgan':
                        r_loss = - tf.reduce_mean(logits_r)
                        f_loss = tf.reduce_mean(logits_f)
                        loss_d = f_loss+r_loss
                        loss_gen = -tf.reduce_mean(logits_f)
                    elif opt.gan_type =='wgangp':
                        r_loss = - tf.reduce_mean(logits_r)
                        f_loss = tf.reduce_mean(logits_f)
                        grad_p = gradient_penalty(D,hr,hr_f)
                        loss_d = f_loss + r_loss + opt.gp_lambda*grad_p
                        loss_gen = -tf.reduce_mean(logits_f)

                if opt.use_perception_loss:
                    p_f = P(hr_f,training=False)
                    p_r = P(hr,training=False)
                    loss_pr = MeanSquaredError()(p_r,p_f)

                loss_g = loss_mse + opt.pl_lambda*loss_pr + opt.gen_lambda*loss_gen


            grad = gen_tape.gradient(loss_g,G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            if epoch>=opt.init_epochs:
                grad_d = disc_tape.gradient(loss_d,D.trainable_weights)
                d_optimizer.apply_gradients(zip(grad_d,D.trainable_weights))
                if opt.gan_type =='wgan':
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
            bar.suffix = "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.6f}, p_loss:{:.6f}, adv:{:.6f},d_f_loss:{:.6f}, d_r_loss:{:.6f},  g_pen:{:.6f}".format(epoch,
                                                                       opt.epochs, step, n_steps_train,
                                                                       time.time() - step_time, loss_mse, loss_pr,
                                                                       loss_gen, f_loss, r_loss, grad_p)
            bar.next()

        bar.finish()
        x_true_train = np.array(x_true_train)
        x_pred_train = np.array(x_pred_train)
        train_mse = np.mean((x_true_train-x_pred_train)**2)
        train_task_score = f1_score(y_true_train,y_pred_train,average='macro')
        print("Epoch: [{}/{}] time:{:.1f}s, mse:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, time.time()-epoch_time,train_mse, train_task_score))

        # Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.init_lr * new_lr_decay)
            print("New learning rate " + str(opt.init_lr * new_lr_decay))

        x_true_test = []
        x_pred_test = []
        y_true_test = np.array([], dtype='int32')
        y_pred_test = np.array([], dtype='int32')
        bar = Bar('>>>', fill='>', max=n_steps_test)
        epoch_time = time.time()
        for step,(lr,hr,y) in enumerate(test_ds):
            step_time = time.time()
            hr_f = G(lr,training=False)
            x_true_test += list(hr.numpy())
            x_pred_test += list(hr_f.numpy())
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(C(hr_f, training=False), axis=1)
            y_true_test = np.append(y_true_test, y_true)
            y_pred_test = np.append(y_pred_test, y_pred)
            bar.suffix= "Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time)
            bar.next()

        bar.finish()
        x_true_test = np.array(x_true_test)
        x_pred_test = np.array(x_pred_test)
        test_mse = np.mean((x_true_test - x_pred_test)**2)
        test_task_score = f1_score(y_true_test, y_pred_test,average='macro')
        print("Testing Epoch: [{}/{}] time:{:.1f}s, mse:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, time.time()-epoch_time,test_mse, test_task_score))
        if test_mse < prev_best:
            G.save(os.path.join(opt.save_dir,'best_gen-'+str(opt.gan_type)+'_'+str(opt.data_type)+'_'+str(opt.sampling_ratio)+'_'+str(opt.use_perception_loss)+'.hdf5'))
            D.save(os.path.join(opt.save_dir,
                                'best_disc-'+str(opt.gan_type)+'_' + str(opt.data_type) + '_' + str(opt.sampling_ratio) + '_' + str(
                                    opt.use_perception_loss) + '.hdf5'))
            print('Saving Best generator with best MSE:'+ str(test_mse))
            prev_best = test_mse
        G.save(os.path.join(opt.save_dir,'last_gen-'+str(opt.gan_type)+'_'+str(opt.data_type)+'_'+str(opt.sampling_ratio)+'_'+str(opt.use_perception_loss)+'.hdf5'))
        D.save(os.path.join(opt.save_dir, 'last_disc-'+str(opt.gan_type)+'_' + str(opt.data_type) + '_' + str(opt.sampling_ratio) + '_' + str(
            opt.use_perception_loss) + '.hdf5'))


# ===============================================================
#                Missing Data Imputation with CNN
# ===============================================================

def train_imp(opt):
    '''
    Training loop for missing data imputation without adversarial loss
    :param opt: Argument Parser
    :return:
    '''
    if opt.data_type == 'ecg':
        inp_shape = (192,2)
        nclasses =  5
        G = imp_model_func('ecg')(inp_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'shl':
        inp_shape = (6,512,2)
        nclasses =  8
        G = imp_model_func('shl')(inp_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'audio':
        inp_shape = (8000,2)
        nclasses =  10
        G = imp_model_func('audio')(inp_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'pam2':
        inp_shape = (27,256,2)
        nclasses =  12
        G = imp_model_func('pam2')(inp_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    print(G.summary())
    lr_v = tf.Variable(opt.init_lr)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)

    train_ds = train_imp_dataset(opt.data_type,  opt.train_batch_size, opt.prob, opt.seed, opt.cont, opt.fixed, opt.shuffle_buffer_size,
                                opt.fetch_buffer_size, opt.resample)
    test_ds = test_imp_dataset(opt.data_type, opt.test_batch_size, opt.prob, opt.seed, opt.cont, opt.fetch_buffer_size)

    prev_best = np.inf
    n_steps_train = len(list(train_ds))
    n_steps_test = len(list(test_ds))

    for epoch in range(opt.epochs):
        x_true_train = []
        x_pred_train = []
        y_true_train = np.array([],dtype='int32')
        y_pred_train = np.array([],dtype='int32')
        bar = Bar('>>>', fill='>', max=n_steps_train)
        epoch_time = time.time()
        for step , (x_m,mask,x,y) in enumerate(train_ds):
            if x.shape[0]<opt.train_batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                x_m_mask = tf.concat([x_m,mask],axis=-1)
                x_pred = G(x_m_mask,training=True)
                if opt.masked_mse_loss:
                    loss_mse = MeanSquaredError()((1 - mask) * x, (1 - mask) * x_pred)/opt.prob
                    x_pred_orig = x_m * mask + x_pred * (1 - mask)
                else:
                    loss_mse = MeanSquaredError()(x, x_pred)
                    x_pred_orig = x_pred
                loss_pr = 0
                if opt.use_perception_loss:
                    p_f = P(x_pred_orig,training=False)
                    p_r = P(x,training=False)
                    loss_pr = MeanSquaredError()(p_r,p_f)
                loss_g = loss_mse + opt.pl_lambda*loss_pr

            grad = tape.gradient(loss_g,G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            x_true_train +=list(x.numpy())
            x_pred_train +=list(x_pred_orig.numpy())
            y_true = np.argmax(y,axis=1)
            y_pred = np.argmax(C(x_pred_orig,training=False),axis=1)
            y_true_train = np.append(y_true_train, y_true)
            y_pred_train = np.append(y_pred_train, y_pred)
            bar.suffix = "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.6f}, p_loss:{:.6f}".format(epoch,
                    opt.epochs,step, n_steps_train,time.time() - step_time, loss_mse,loss_pr)
            bar.next()

        bar.finish()
        x_true_train = np.array(x_true_train)
        x_pred_train = np.array(x_pred_train)
        train_mse = np.mean((x_true_train-x_pred_train)**2)
        train_task_score = f1_score(y_true_train,y_pred_train,average='macro')
        print("Epoch: [{}/{}] time:{:.1f}s, mse:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, time.time()-epoch_time,train_mse, train_task_score))

        # Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.init_lr * new_lr_decay)
            print("New learning rate" + str(opt.init_lr * new_lr_decay))

        x_true_test = []
        x_pred_test = []
        y_true_test = np.array([], dtype='int32')
        y_pred_test = np.array([], dtype='int32')
        bar = Bar('>>>', fill='>', max=n_steps_test)
        epoch_time = time.time()
        for step,(x_m,mask,x,y) in test_ds:
            step_time = time.time()
            x_m_mask = tf.concat([x_m,mask],axis=-1)
            x_pred = G(x_m_mask,training=False)
            if opt.masked_mse_loss:
                x_pred_orig = x_m*mask+x_pred*(1-mask)
            else:
                x_pred_orig = x_pred
            x_true_test += list(x.numpy())
            x_pred_test += list(x_pred_orig.numpy())
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(C(x_pred_orig, training=False), axis=1)
            y_true_test = np.append(y_true_test, y_true)
            y_pred_test = np.append(y_pred_test, y_pred)
            bar.suffix = "Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time)
            bar.next()

        bar.finish()
        x_true_test = np.array(x_true_test)
        x_pred_test = np.array(x_pred_test)
        test_mse = np.mean((x_true_test- x_pred_test)**2)
        test_task_score= f1_score(y_true_test, y_pred_test,average='macro')
        print("Testing Epoch: [{}/{}] time:{:.1f}s, mse:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs,time.time()-epoch_time, test_mse, test_task_score))
        if opt.cont:
            if opt.fixed:
                str_imp = 'imp-cont-fixed_'
            else:
                str_imp = 'imp-cont_'
        else:
            if opt.fixed:
                str_imp = 'imp-fixed_'
            else:
                str_imp = 'imp_'
        if test_mse < prev_best:
            G.save(os.path.join(opt.save_dir,'best_cnn-'+str_imp+str(opt.data_type)+'_'+str(opt.prob)+'_'+str(opt.use_perception_loss)+'_'+str(opt.masked_mse_loss)+'.hdf5'))
            print('Saving Best generator with best MSE:' + str(test_mse))
            prev_best = test_mse
        G.save(os.path.join(opt.save_dir,'last_cnn-'+str_imp+str(opt.data_type)+'_'+str(opt.prob)+'_'+str(opt.use_perception_loss)+'_'+str(opt.masked_mse_loss)+'.hdf5'))


# ===============================================================
#                Missing Data Imputation with GAN
# ===============================================================

def train_imp_gan(opt):
    '''
    Training loop for missing data imputation with adversarial loss
    :param opt: Argument Parser
    :return:
    '''
    if opt.data_type == 'ecg':
        inp_shape = (192,2)
        inp_disc_shape = (192,1)
        nclasses =  5
        G = imp_model_func('ecg')(inp_shape)
        D = disc_model_func('ecg')(inp_disc_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'shl':
        inp_shape = (6,512,2)
        inp_disc_shape = (6,512,1)
        nclasses =  5
        G = imp_model_func('shl')(inp_shape)
        D = disc_model_func('shl')(inp_disc_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'audio':
        inp_shape = (8000,2)
        inp_disc_shape = (8000,1)
        nclasses =  10
        G = imp_model_func('audio')(inp_shape)
        D = disc_model_func('audio')(inp_disc_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    elif opt.data_type == 'pam2':
        inp_shape = (27,256,2)
        inp_disc_shape = (27,256,1)
        nclasses =  12
        G = imp_model_func('pam2')(inp_shape)
        D = disc_model_func('pam2')(inp_disc_shape)
        C = load_model(opt.classifier_path)
        if opt.use_perception_loss:
            P = Model(inputs = C.input, outputs = C.layers[-3].output)

    print(G.summary())
    print(D.summary())

    lr_v = tf.Variable(opt.init_lr)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=opt.beta1)

    train_ds = train_imp_dataset(opt.data_type,  opt.train_batch_size, opt.prob, opt.seed, opt.cont, opt.fixed, opt.shuffle_buffer_size,
                                opt.fetch_buffer_size, opt.resample)
    test_ds = test_imp_dataset(opt.data_type, opt.test_batch_size, opt.prob, opt.seed, opt.cont, opt.fetch_buffer_size)

    prev_best = np.inf
    n_steps_train = len(list(train_ds))
    n_steps_test = len(list(test_ds))

    for epoch in range(opt.epochs):
        x_true_train = []
        x_pred_train = []
        y_true_train = np.array([],dtype='int32')
        y_pred_train = np.array([],dtype='int32')
        bar = Bar('>>>', fill='>', max=n_steps_train)
        epoch_time = time.time()
        for step , (x_m,mask,x,y) in enumerate(train_ds):
            if x.shape[0]<opt.train_batch_size:
                break
            step_time = time.time()
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                x_m_mask = tf.concat([x_m,mask],axis=-1)
                x_pred = G(x_m_mask,training=True)
                if opt.masked_mse_loss:
                    loss_mse = MeanSquaredError()((1-mask)*x,(1-mask)*x_pred)/opt.prob
                    x_pred_orig = x_m*mask +x_pred*(1-mask)
                else:
                    loss_mse = MeanSquaredError()(x, x_pred)
                    x_pred_orig = x_pred
                loss_gen = 0
                f_loss = 0
                r_loss = 0
                loss_d = 0
                loss_pr = 0
                grad_p = 0
                if epoch >= opt.init_epochs:
                    logits_f = D(x_pred_orig, training=True)
                    logits_r = D(x, training=True)
                    if opt.gan_type == 'normal':
                        f_loss = MeanSquaredError()(tf.zeros_like(logits_f), logits_f)
                        r_loss = MeanSquaredError()(tf.ones_like(logits_r), logits_r)
                        loss_d = f_loss + r_loss
                        loss_gen = MeanSquaredError()(tf.ones_like(logits_f), logits_f)
                    elif opt.gan_type == 'normalls':
                        f_loss = MeanSquaredError()(
                            tf.zeros_like(logits_f) + tf.random.uniform(logits_f.shape, 0, 1) * 0.3,
                            logits_f)  # Label Smoothing
                        r_loss = MeanSquaredError()(
                            tf.ones_like(logits_r) - 0.3 + tf.random.uniform(logits_f.shape, 0, 1) * 0.5,
                            logits_r)  # Label Smoothing
                        loss_d = f_loss + r_loss
                        loss_gen = MeanSquaredError()(
                            tf.ones_like(logits_f) - 0.3 + tf.random.uniform(logits_f.shape, 0, 1) * 0.5,
                            logits_f)  # Label Smoothing
                    elif opt.gan_type == 'wgan':
                        r_loss = - tf.reduce_mean(logits_r)
                        f_loss = tf.reduce_mean(logits_f)
                        loss_d = f_loss + r_loss
                        loss_gen = -tf.reduce_mean(logits_f)
                    elif opt.gan_type == 'wgangp':
                        r_loss = - tf.reduce_mean(logits_r)
                        f_loss = tf.reduce_mean(logits_f)
                        grad_p = gradient_penalty(D, x, x_pred_orig)
                        loss_d = f_loss + r_loss + opt.gp_lambda * grad_p
                        loss_gen = -tf.reduce_mean(logits_f)

                if opt.use_perception_loss:
                    p_f = P(x_pred_orig,training=False)
                    p_r = P(x,training=False)
                    loss_pr = MeanSquaredError()(p_r,p_f)

                loss_g = loss_mse + opt.pl_lambda*loss_pr + opt.gen_lambda*loss_gen

            grad = gen_tape.gradient(loss_g,G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            if epoch>=opt.init_epochs:
                grad_d = disc_tape.gradient(loss_d,D.trainable_weights)
                d_optimizer.apply_gradients(zip(grad_d,D.trainable_weights))
                if opt.gan_type =='wgan':
                    for l in D.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -opt.clip_value, opt.clip_value) for w in weights]
                        l.set_weights(weights)

            x_true_train +=list(x.numpy())
            x_pred_train +=list(x_pred_orig.numpy())
            y_true = np.argmax(C(x,training=False),axis=1)
            y_pred = np.argmax(C(x_pred_orig,training=False),axis=1)
            y_true_train = np.append(y_true_train, y_true)
            y_pred_train = np.append(y_pred_train, y_pred)
            bar.suffix = "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse:{:.6f}, p_loss:{:.6f}, adv:{:.6f}, d_f_loss:{:.6f}, d_r_loss:{:.6f}, g_pen:{:.6f}".format(epoch,
                                                                       opt.epochs, step, n_steps_train,
                                                                       time.time() - step_time, loss_mse, loss_pr,
                                                                       loss_gen, f_loss, r_loss, grad_p)
            bar.next()

        bar.finish()
        x_true_train = np.array(x_true_train)
        x_pred_train = np.array(x_pred_train)
        train_mse = np.mean((x_true_train-x_pred_train)**2)
        train_task_score = f1_score(y_true_train,y_pred_train,average='macro')
        print("Epoch: [{}/{}] time:{:.1f}s, mse:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, time.time()-epoch_time,train_mse, train_task_score))

        # Update Learning Rate
        if epoch != 0 and (epoch % opt.decay_every == 0):
            new_lr_decay = opt.lr_decay ** (epoch // opt.decay_every)
            lr_v.assign(opt.init_lr * new_lr_decay)
            print("New learning rate " +str(opt.init_lr * new_lr_decay))

        x_true_test = []
        x_pred_test = []
        y_true_test = np.array([], dtype='int32')
        y_pred_test = np.array([], dtype='int32')
        bar = Bar('>>>', fill='>', max=n_steps_test)
        epoch_time = time.time()
        for step,(x_m,mask,x,y) in enumerate(test_ds):
            step_time = time.time()
            x_m_mask = tf.concat([x_m,mask],axis=-1)
            x_pred = G(x_m_mask,training=False)
            if opt.masked_mse_loss:
                x_pred_orig = x_m*mask + x_pred*(1-mask)
            else:
                x_pred_orig = x_pred
            x_true_test += list(x.numpy())
            x_pred_test += list(x_pred_orig.numpy())
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(C(x_pred_orig, training=False), axis=1)
            y_true_test = np.append(y_true_test, y_true)
            y_pred_test = np.append(y_pred_test, y_pred)
            bar.suffix = "Testing: Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s".format(
                epoch, opt.epochs, step, n_steps_test, time.time() - step_time)
            bar.next()

        bar.finish()
        x_true_test = np.array(x_true_test)
        x_pred_test = np.array(x_pred_test)
        test_mse = np.mean((x_true_test- x_pred_test)**2)
        test_task_score= f1_score(y_true_test, y_pred_test,average='macro')
        print("Testing Epoch: [{}/{}]  time:{:.1f}s, mse:{:.6f}, f1_score:{:.6f} ".format(
            epoch, opt.epochs, time.time()-epoch_time, test_mse, test_task_score))
        if opt.cont:
            if opt.fixed:
                str_imp = 'imp-cont-fixed-'
            else:
                str_imp = 'imp-cont-'
        else:
            if opt.fixed:
                str_imp = 'imp-fixed-'
            else:
                str_imp = 'imp-'
        if test_mse < prev_best:
            G.save(os.path.join(opt.save_dir,'best_gen-'+str_imp+str(opt.gan_type)+'_'+str(opt.data_type)+'_'+str(opt.prob)+'_'+str(opt.use_perception_loss)+'_'+str(opt.masked_mse_loss)+'.hdf5'))
            D.save(os.path.join(opt.save_dir,
                                'best_disc-'+str_imp +str(opt.gan_type)+'_'+ str(opt.data_type) + '_' + str(opt.prob) + '_' + str(
                                    opt.use_perception_loss)+'_'+str(opt.masked_mse_loss) + '.hdf5'))
            print('Saving Best generator with best MSE:'+str(test_mse))
            prev_best = test_mse
        G.save(os.path.join(opt.save_dir,'last_gen-'+str_imp+str(opt.gan_type)+'_'+str(opt.data_type)+'_'+str(opt.prob)+'_'+str(opt.use_perception_loss)+'_'+str(opt.masked_mse_loss)+'.hdf5'))
        D.save(os.path.join(opt.save_dir,
                            'last_disc-'+str_imp+str(opt.gan_type)+'_' + str(opt.data_type) + '_' + str(opt.prob) + '_' + str(
                                opt.use_perception_loss)+'_'+str(opt.masked_mse_loss) + '.hdf5'))










