import urllib.request
import os
import zipfile

print('Beginning file download with urllib2...')

dirs = ['Raw_data','Raw_data/train','Raw_data/test']

for d in dirs:
    try:
        os.makedirs(d)
    except FileExistsError:  # directory already exists
        pass

# Download Train Files
base_url = 'http://www.shl-dataset.org/wp-content/uploads/SHLChallenge/'
train_files = ['train_acc.zip', 'train_gyr.zip', 'train_mag.zip', 'train_lacc.zip', 'train_gra.zip', 'train_ori.zip', 'train_pressure.zip','train_label.zip']
train_files_txt = ['Acc_x.txt', 'Gyr_x.txt', 'Mag_x.txt', 'LAcc_x.txt', 'Gra_x.txt', 'Ori_x.txt', 'Pressure.txt','Label.txt']

#train_files = ['train_pressure.zip']

for file_name, file_name_txt in zip(train_files, train_files_txt):
    url = base_url + file_name
    file_path = os.path.join('Raw_data/train', file_name)
    file_txt = os.path.join('Raw_data/train', file_name_txt)

    if os.path.exists(file_txt):
        print('{} exists'.format(file_name_txt))
    else:
        print('Downloading {}'.format(file_name))
        urllib.request.urlretrieve(url, file_path)
        with zipfile.ZipFile(file_path, 'r') as file:
            print('Extracting {}'.format(file_name))
            file.extractall('Raw_data')
        os.remove(file_path)

# Download Train Order
file_name = 'train_order.zip'
file_name_txt = 'train_order.txt'
url = 'http://www.shl-dataset.org/wp-content/uploads/2018/06/train_order.zip'
file_path = os.path.join('Raw_data/train', file_name)
file_txt = os.path.join('Raw_data/train', file_name_txt)

if os.path.exists(file_txt):
    print('{} exists'.format(file_name_txt))
else:
    print('Downloading {}'.format(file_name))
    urllib.request.urlretrieve(url, file_path)
    with zipfile.ZipFile(file_path, 'r') as file:
        file.extractall('Raw_data/train')
    os.remove(file_path)

# Download Test Files

base_url = 'http://www.shl-dataset.org/wp-content/uploads/SHLChallenge/'
test_files = ['test_acc.zip', 'test_gyr.zip', 'test_mag.zip', 'test_lacc.zip', 'test_gra.zip', 'test_ori.zip', 'test_pressure.zip']
test_files_txt = ['Acc_x.txt', 'Gyr_x.txt', 'Mag_x.txt', 'LAcc_x.txt', 'Gra_x.txt', 'Ori_x.txt', 'Pressure.txt']

for file_name, file_name_txt in zip(test_files, test_files_txt):
    url = base_url + file_name
    file_path = os.path.join('Raw_data/test', file_name)
    file_txt = os.path.join('Raw_data/test', file_name_txt)

    if os.path.exists(file_txt):
        print('{} exists'.format(file_name_txt))
    else:
        print('Downloading {}'.format(file_name))
        urllib.request.urlretrieve(url, file_path)
        with zipfile.ZipFile(file_path, 'r') as file:
            print('Extracting {}'.format(file_name))
            file.extractall('Raw_data')
        os.remove(file_path)


# Delete Zip Files
# dir_name = 'Raw_data/train'
# files = os.listdir(dir_name)

# for item in files:
#     if item.endswith(".zip"):
#         os.remove(os.path.join(dir_name, item))

# dir_name = 'Raw_data/test'
# files = os.listdir(dir_name)

# for item in files:
#     if item.endswith(".zip"):
#         os.remove(os.path.join(dir_name, item))