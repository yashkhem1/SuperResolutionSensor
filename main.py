from src.parsing import Options
from src.train import train_clf, train_sr, train_sr_gan, train_imp, train_imp_gan
from src.evaluate import evaluate_clf, evaluate_ecg_sr, evaluate_ecg_imp, evaluate_shl_imp, evaluate_shl_sr
import os

if __name__ == "__main__":
    opt = Options().parse()
    if opt.decay_half:
        opt.decay_every = opt.epochs//2
    else:
        opt.decay_every = opt.epochs+1 #This will ensure that the learning rate does not change
    if opt.save_dir:
        os.makedirs(opt.save_dir,exist_ok=True)

    if opt.evaluate:
        if opt.model_type == 'clf':
            evaluate_clf(opt)

        elif opt.model_type in ['sr','sr_gan']:
            if opt.data_type=='ecg':
                evaluate_ecg_sr(opt)

            elif opt.data_type=='shl':
                evaluate_shl_sr(opt)

        elif opt.model_type in ['imp','imp_gan']:
            if opt.data_type=='ecg':
                evaluate_ecg_imp(opt)

            elif opt.data_type=='shl':
                evaluate_shl_imp(opt)

    elif opt.model_type == 'clf':
        train_clf(opt)

    elif opt.model_type == 'sr':
        train_sr(opt)

    elif opt.model_type == 'sr_gan':
        train_sr_gan(opt)

    elif opt.model_type == 'imp':
        train_imp(opt)

    elif opt.model_type == 'imp_gan':
        train_imp_gan(opt)





