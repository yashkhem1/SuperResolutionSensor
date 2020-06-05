from src.models import *
from src.data_loader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix

def evaluate_clf(opt):
    C = load_model(opt.classifier_path)
    test_X,test_Y = read_test_data(opt.data_type)
    if opt.data_type=='ecg':
        test_X = test_X[:,::opt.sampling_ratio,:]
    if opt.use_sr_clf:
        G = load_model(opt.model_path)
        test_X = G(test_X,training=False).numpy()
    y_true = np.argmax(test_Y,axis=1)
    y_pred = np.argmax(C(test_X,training=False),axis=1)
    accuracy = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred,average='minor')
    cf_matrix = confusion_matrix(y_true,y_pred)
    print('Accuracy: ', accuracy)
    print('F1_score: ', f1)
    print('Confusion Matrix: ', cf_matrix)

def evaluate_ecg_sr(opt):
    G = load_model(opt.model_path)
    C = load_model(opt.classifier_path)
    test_X,test_Y = read_test_data(opt.data_type)
    x_true = test_X
    x_pred = G(x_true[:, ::opt.sampling_ratio, :],training=False).numpy()
    y_true = np.argmax(test_Y,axis=1)
    y_pred_sr = np.argmax(C(x_pred,training=False),axis=1)
    y_pred_hr= np.argmax(C(x_true,training=False),axis=1)
    print('MSE: ', np.mean((x_true-x_pred)**2))
    print('Accuracy HR: ', accuracy_score(y_true,y_pred_hr))
    print('Accuracy SR: ', accuracy_score(y_true, y_pred_sr))
    print('F1_score HR: ', f1_score(y_true,y_pred_hr, average='minor'))
    print('F1_score SR: ', f1_score(y_true, y_pred_sr, average='minor'))
    print('Confusion Matrix HR: ', confusion_matrix(y_true,y_pred_hr))
    print('Confusion Matrix SR: ', confusion_matrix(y_true, y_pred_sr))