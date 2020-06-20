import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.data_loader import *
import sys

def plot_ecg_sr(sampling_ratio, model_directory='ckpt',sample_number=None):
    test_X, _ = read_test_data('ecg')
    if sample_number:
        sample = test_X[sample_number]
    else:
        sample_number = np.random.randint(test_X.shape[0])
        sample = test_X[sample_number]

    model_list = ['best_cnn_ecg_'+str(sampling_ratio)+'_0.pt','best_cnn_ecg_'+str(sampling_ratio)+'_1.pt',
                  'best_gen-wgangp_ecg_'+str(sampling_ratio)+'_0.pt','best_gen-wgangp_ecg_'+str(sampling_ratio)+'_1.pt' ]

    sample_hr = sample.copy().reshape(-1)
    sample_list = [sample_hr]
    for path in model_list:
        G = load_model(os.path.join(model_directory,path))
        sample_sr = G.predict(sample[::sampling_ratio, :].reshape(1,-1,1),batch_size=128,verbose=1)
        sample_list.append(sample_sr.reshape(-1))

    color_list = ['k','r','b','g','y']
    plt.figure(figsize=(15, 10))
    for i,sample in enumerate(sample_list):
        plt.plot(sample, color_list[i])
    plt.legend(['true', 'cnn w/o p_loss', 'cnn with p_loss', 'gan w/o p_loss', 'gan with p_loss'])
    # plt.show()
    plt.savefig('ecg_'+ str(sample_number)+'_'+str(sampling_ratio)+'.png')

    color_list = ['r','b','g','y']
    file_names = ['cnn','cnn_p','gen','gen_p']
    legends = ['cnn w/o p_loss', 'cnn with p_loss', 'gan w/o p_loss', 'gan with p_loss']
    sample_list = sample_list[1:]
    for i,sample in enumerate(sample_list):
        plt.figure(figsize=(15, 10))
        plt.plot(sample_hr,'k')
        plt.plot(sample,color_list[i])
        plt.legend(['true',legends[i]])
        plt.savefig('ecg_'+file_names[i]+'_'+str(sample_number)+'_'+str(sampling_ratio)+'.png')


def plot_ecg_imp(prob,seed,cont,model_directory='ckpt',sample_number=None):
    np.random.seed(seed)
    test_X, _ = read_test_data('ecg')
    if sample_number:
        sample = test_X[sample_number]
    else:
        sample_number = np.random.randint(test_X.shape[0])
        sample = test_X[sample_number]

    indices = np.arange(192)
    n_missing = int(prob * 192)
    test_X_m = np.zeros(sample.shape)
    test_mask = np.ones(sample.shape)
    if cont:
        missing_start = np.random.randint(0, int((1 - prob) * 192) + 1)
        missing_indices = np.arange(missing_start, missing_start + n_missing)
    else:
        missing_indices = np.random.choice(indices, n_missing, replace=False)
    test_X_m = sample
    test_X_m[missing_indices] = 0
    test_mask[missing_indices] = 0
    test_X_m_mask = np.concatenate([test_X_m,test_mask],axis=-1).reshape(1,-1,2)


    if cont:
        model_list = ['best_cnn-imp-cont_ecg_' + str(prob) + '_0_1.pt', 'best_cnn-imp-cont_ecg_' + str(prob) + '_1_1.pt',
                      'best_gen-imp-cont-wgangp_ecg_' + str(prob) + '_0_1.pt',
                      'best_gen-imp-cont-wgangp_ecg_' + str(prob) + '_1_1.pt']
    else:
        model_list = ['best_cnn-imp_ecg_' + str(prob) + '_0_1.pt', 'best_cnn-imp_ecg_' + str(prob) + '_1_1.pt',
                      'best_gen-imp-wgangp_ecg_' + str(prob) + '_0_1.pt',
                      'best_gen-imp-wgangp_ecg_' + str(prob) + '_1_1.pt']

    sample_orig = sample.copy()
    sample_list = [sample_orig]
    for path in model_list:
        G = load_model(os.path.join(model_directory,path))
        sample_imp = G.predict(test_X_m_mask,batch_size=128,verbose=1)
        sample_imp = test_X_m*test_mask + sample_imp*(1-test_mask)
        sample_list.append(sample_imp.reshape(-1))

    color_list = ['k', 'r', 'b', 'g', 'y']
    plt.figure(figsize=(15, 10))
    for i, sample in enumerate(sample_list):
        plt.plot(sample, color_list[i])

    plt.legend(['true', 'cnn w/o p_loss', 'cnn with p_loss', 'gan w/o p_loss', 'gan with p_loss'])
    # plt.show()
    if cont:
        plt.savefig('ecg_imp-cont_' + str(sample_number) + '_' + str(prob) + '.png')
    else:
        plt.savefig('ecg_imp_' + str(sample_number) + '_' + str(prob) + '.png')

    color_list = ['r', 'b', 'g', 'y']
    file_names = ['cnn', 'cnn_p', 'gen', 'gen_p']
    legends = ['cnn w/o p_loss', 'cnn with p_loss', 'gan w/o p_loss', 'gan with p_loss']
    sample_list = sample_list[1:]
    for i, sample in enumerate(sample_list):
        plt.figure(figsize=(15, 10))
        plt.plot(sample_orig, 'k')
        plt.plot(sample, color_list[i])
        plt.legend(['true', legends[i]])
        if cont:
            plt.savefig('ecg_imp-cont_' + file_names[i] + '_' + str(sample_number) + '_' + str(prob) + '.png')
        else:
            plt.savefig('ecg_imp_' + file_names[i] + '_' + str(sample_number) + '_' + str(prob) + '.png')

if __name__=="__main__":
    plot_type = sys.argv[1]
    if plot_type == 'sr':
        sampling_ratio = int(sys.argv[2])
        sample_number = None
        if len(sys.argv)==4:
            sample_number = int(sys.argv[3])
        plot_ecg_sr(sampling_ratio,'ckpt',sample_number)

    elif plot_type=='imp':
        prob = int(sys.argv[2])
        seed = int(sys.argv[3])
        cont = int(sys.argv[4])
        sample_number = None
        if len(sys.argv)==6:
            sample_number = int(sys.argv[5])
        plot_ecg_imp(prob,seed,cont,'ckpt',sample_number)


