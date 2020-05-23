from src.parsing import Options
from src.train import train_clf, train_sr

if __name__ == "__main__":
    opt = Options().parse()
    opt.decay_every = opt.epochs//2
    if opt.model_type == 'clf':
        train_clf(opt)

    elif opt.model_type == 'sr':
        train_sr(opt)


