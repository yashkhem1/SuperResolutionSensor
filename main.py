from src.parsing import Options
from src.train import train_clf

if __name__ == "__main__":
    opt = Options().parse()
    opt.decay_every = opt.epochs//2
    if opt.model_type == 'clf':
        train_clf(opt)


