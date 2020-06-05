import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.data_loader import *
import sys

def plot_ecg(sampling_ratio, model_directory='ckpt',sample_number=None):
    test_X, _ = read_test_data('ecg')
    if sample_number:
        sample = test_X[sample_number]
    else:
        sample_number = np.random.randint(test_X.shape[0])
        sample = test_X[sample_number]

    model_list = ['best_cnn_ecg_'+str(sampling_ratio)+'_0.pt','best_cnn_ecg_'+str(sampling_ratio)+'_1.pt',
                  'best_gen_ecg_'+str(sampling_ratio)+'_0.pt','best_gen_ecg_'+str(sampling_ratio)+'_1.pt' ]

    sample_hr = sample.copy().reshape(-1)
    sample_list = [sample_hr]
    for path in model_list:
        G = load_model(os.path.join(model_directory,path))
        sample_sr = G(sample[::sampling_ratio, :].reshape(1,-1,1), training=False).numpy()
        sample_list.append(sample_sr.reshape(-1))

    color_list = ['k','r','b','g','y']
    plt.figure(figsize=(15, 10))
    for i,sample in enumerate(sample_list):
        plt.plot(sample, color_list[i])
    plt.legend(['true', 'cnn w/o p_loss', 'cnn with p_loss', 'gan w/o p_loss', 'gan with p_loss'])
    # plt.show()
    plt.savefig('ecg_'+ str(sample_number)+'_'+str(sampling_ratio)+'.png')

if __name__=="__main__":
    sampling_ratio = int(sys.argv[1])
    model_directory = sys.argv[2]
    sample_number = None
    if len(sys.argv)==4:
        sample_number = int(sys.argv[3])
    plot_ecg(sampling_ratio,model_directory,sample_number)

