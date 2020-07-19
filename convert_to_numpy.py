import numpy as np
from scipy.stats import mode
import os

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.shape[0]-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L,a.shape[1] ), strides=(S*n,n,a.strides[1]))

''' This is going to take up a lot of memory. Close all the other background applications before running '''
### Save train_data

data_path='Raw_data/train'

# Acc data files
accx_path = data_path+'/Acc_x.txt'
accy_path = data_path+'/Acc_y.txt'
accz_path = data_path+'/Acc_z.txt'

# Gyro data files
gyrox_path = data_path+'/Gyr_x.txt'
gyroy_path = data_path+'/Gyr_y.txt'
gyroz_path = data_path+'/Gyr_z.txt'

# Labels
label_path = data_path+'/Label.txt'

# Training dataframe order
order_path = data_path+'/train_order.txt'

print('Loading acc')
accx =  np.loadtxt(accx_path)
accy =  np.loadtxt(accy_path)
accz =  np.loadtxt(accz_path)

print('Loading gyro')
gyrox =  np.loadtxt(gyrox_path)
gyroy =  np.loadtxt(gyroy_path)
gyroz =  np.loadtxt(gyroz_path)

print('Loading labels')
label= np.loadtxt(label_path)
print('Loading order')
order = np.loadtxt(order_path)

ordering = np.argsort(order)
accx = accx[ordering]
accy = accy[ordering]
accz = accz[ordering]

gyrox = gyrox[ordering]
gyroy = gyroy[ordering]
gyroz = gyroz[ordering]

label = label[ordering]

save_path = 'Extracted_data'
np.savez(save_path+'/Train_data', accx,accy,accz,gyrox,gyroy,gyroz)
np.save(save_path+'/Train_labels', label)

X_train_prev = np.float32(np.dstack((accx, accy, accz, gyrox, gyroy, gyroz)))
y_train_prev = np.expand_dims(np.float32(label),axis=-1)

X_train = []
y_train = []
for i in range(X_train_prev.shape[0]):
    X_train.append(strided_app(X_train_prev[i],512,250))
    y_train.append(strided_app(y_train_prev[i],512,250))

del X_train_prev, y_train_prev

X_train,y_train = np.asarray(X_train), np.asarray(y_train)
X_train = X_train.reshape(-1,512,6)
y_train,_ = mode(y_train.reshape(-1,512),axis=1)
y_train = np.asarray(y_train)
np.save('data/shl_train_x.npy',X_train)
np.save('data/shl_train_y.npy',y_train)


del X_train, y_train
## Save test_data

data_path='Raw_data/test'

# Acc data files
accx_path = data_path+'/Acc_x.txt'
accy_path = data_path+'/Acc_y.txt'
accz_path = data_path+'/Acc_z.txt'

# Gyro data files
gyrox_path = data_path+'/Gyr_x.txt'
gyroy_path = data_path+'/Gyr_y.txt'
gyroz_path = data_path+'/Gyr_z.txt'

# Labels
label_path = data_path+'/Label.txt'

# Training dataframe order
order_path = data_path+'/test_order.txt'

print('Loading acc')
accx =  np.loadtxt(accx_path)
accy =  np.loadtxt(accy_path)
accz =  np.loadtxt(accz_path)

print('Loading gyro')
gyrox =  np.loadtxt(gyrox_path)
gyroy =  np.loadtxt(gyroy_path)
gyroz =  np.loadtxt(gyroz_path)

print('Loading labels')
label= np.loadtxt(label_path)
print('Loading order')
order = np.loadtxt(order_path)

ordering = np.argsort(order)
accx = accx[ordering]
accy = accy[ordering]
accz = accz[ordering]

gyrox = gyrox[ordering]
gyroy = gyroy[ordering]
gyroz = gyroz[ordering]

label = label[ordering]
#


X_test_prev = np.float32(np.dstack((accx, accy, accz, gyrox, gyroy, gyroz)))
y_test_prev = np.expand_dims(np.float32(label),axis=-1)

X_test = []
y_test = []
for i in range(X_test_prev.shape[0]):
    X_test.append(strided_app(X_test_prev[i],512,250))
    y_test.append(strided_app(y_test_prev[i],512,250))

del X_test_prev, y_test_prev

X_test,y_test = np.asarray(X_test), np.asarray(y_test)
X_test = X_test.reshape(-1,512,6)
y_test,_ = mode(y_test.reshape(-1,512),axis=1)
y_test = np.asarray(y_test)
np.save('data/shl_test_x.npy',X_test)
np.save('data/shl_test_y.npy',y_test)

del X_test, y_test


## Intermediate PreProcessing
# save_path = 'Extracted_data'
# np.savez(save_path+'/Test_data', accx,accy,accz,gyrox,gyroy,gyroz)
# np.save(save_path+'/Test_labels', label)

# train_data = np.load('Extracted_data/Test_data.npz')
# label = np.load('Extracted_data/Test_labels.npy')
# print('loaded_data')
# accx = train_data['arr_0']
# accy = train_data['arr_1']
# accz = train_data['arr_2']
# gyrox = train_data['arr_3']
# gyroy = train_data['arr_4']
# gyroz = train_data['arr_5']






