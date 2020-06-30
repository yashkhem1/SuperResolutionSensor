from src.models import *
from src.data_loader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix

def evaluate_clf(opt):
    C = load_model(opt.classifier_path)
    test_X,test_Y = read_test_data(opt.data_type)
    if opt.data_type=='ecg':
        if opt.sampling_ratio != 1:
            test_X = test_X[:,::opt.sampling_ratio,:]
            if opt.use_sr_clf:
                G = load_model(opt.model_path)
                test_X = G.predict(test_X,batch_size=opt.test_batch_size,verbose=1)

        if opt.prob != 0:
            np.random.seed(opt.seed)
            indices = np.arange(192)
            n_missing = int(opt.prob * 192)
            test_X_m = np.zeros(test_X.shape)
            test_mask = np.ones(test_X.shape)
            for i, data in enumerate(test_X):
                if opt.cont:
                    missing_start = np.random.randint(0, int((1 - opt.prob) * 192) + 1)
                    missing_indices = np.arange(missing_start, missing_start + n_missing)
                else:
                    missing_indices = np.random.choice(indices, n_missing, replace=False)
                test_X_m[i] = data
                test_X_m[i][missing_indices] = 0
                test_mask[i][missing_indices] = 0

            if opt.use_imp_clf:
                G = load_model(opt.model_path)
                test_X_m_mask = np.concatenate([test_X_m, test_mask], axis=-1)
                x_pred = G.predict(test_X_m_mask, batch_size=opt.test_batch_size, verbose=1)
                test_X = test_X_m * test_mask + x_pred * (1 - test_mask)

            else:
                test_X = test_X_m

    y_true = np.argmax(test_Y,axis=1)
    y_pred = np.argmax(C.predict(test_X,batch_size=opt.test_batch_size,verbose=1),axis=1)
    accuracy = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred,average='macro')
    cf_matrix = confusion_matrix(y_true,y_pred)
    print('Accuracy: ', accuracy)
    print('F1_score: ', f1)
    print('Confusion Matrix: ', cf_matrix)

def evaluate_ecg_sr(opt):
    G = load_model(opt.model_path)
    C = load_model(opt.classifier_path)
    test_X,test_Y = read_test_data(opt.data_type)
    x_true = test_X
    x_pred = G.predict(x_true[:, ::opt.sampling_ratio, :],batch_size=opt.test_batch_size,verbose=1)
    y_true = np.argmax(test_Y,axis=1)
    y_pred_sr = np.argmax(C.predict(x_pred,batch_size=opt.test_batch_size,verbose=1),axis=1)
    y_pred_hr= np.argmax(C.predict(x_true,batch_size=opt.test_batch_size,verbose=1),axis=1)
    print('MSE: ', np.mean((x_true-x_pred)**2))
    print('Accuracy HR: ', accuracy_score(y_true,y_pred_hr))
    print('Accuracy SR: ', accuracy_score(y_true, y_pred_sr))
    print('F1_score HR: ', f1_score(y_true,y_pred_hr, average='macro'))
    print('F1_score SR: ', f1_score(y_true, y_pred_sr, average='macro'))
    print('Confusion Matrix HR: ', confusion_matrix(y_true,y_pred_hr))
    print('Confusion Matrix SR: ', confusion_matrix(y_true, y_pred_sr))

def evaluate_ecg_imp(opt):
    G = load_model(opt.model_path)
    C = load_model(opt.classifier_path)
    test_X, test_Y = read_test_data(opt.data_type)
    x_true = test_X
    np.random.seed(opt.seed)
    indices = np.arange(192)
    n_missing = int(opt.prob * 192)
    test_X_m = np.zeros(test_X.shape)
    test_mask = np.ones(test_X.shape)
    for i, data in enumerate(test_X):
        if opt.cont:
            missing_start = np.random.randint(0, int((1 - opt.prob) * 192) + 1)
            missing_indices = np.arange(missing_start, missing_start + n_missing)
        else:
            missing_indices = np.random.choice(indices, n_missing, replace=False)
        test_X_m[i] = data
        test_X_m[i][missing_indices] = 0
        test_mask[i][missing_indices] = 0

    test_X_m_mask = np.concatenate([test_X_m,test_mask],axis=-1)
    x_pred = G.predict(test_X_m_mask,batch_size=opt.test_batch_size,verbose=1)
    x_pred = test_X_m*test_mask + x_pred*(1-test_mask)
    y_true = np.argmax(test_Y, axis=1)
    y_pred_imp = np.argmax(C.predict(x_pred, batch_size=opt.test_batch_size, verbose=1), axis=1)
    y_pred_orig = np.argmax(C.predict(x_true, batch_size=opt.test_batch_size, verbose=1), axis=1)
    print('MSE: ', np.mean((x_true - x_pred) ** 2))
    print('Accuracy Orig: ', accuracy_score(y_true, y_pred_orig))
    print('Accuracy Imp: ', accuracy_score(y_true, y_pred_imp))
    print('F1_score Orig: ', f1_score(y_true, y_pred_orig, average='macro'))
    print('F1_score Imp: ', f1_score(y_true, y_pred_imp, average='macro'))
    print('Confusion Matrix HR: ', confusion_matrix(y_true, y_pred_orig))
    print('Confusion Matrix SR: ', confusion_matrix(y_true, y_pred_imp))

