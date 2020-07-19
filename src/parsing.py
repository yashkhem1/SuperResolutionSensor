import argparse
from pprint import pprint

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_type' , type=str, help='Dataset Type')
        self.parser.add_argument('--init_lr', type=float,  default=0.0001 , help='Initial Learning Rate for the model')
        self.parser.add_argument('--epochs', type=float, default=200, help='Number of epochs in training')
        self.parser.add_argument('--lr_decay', type=float, default=0.1, help='Decay factor of learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='Value of Beta1 for adam optimizer')
        self.parser.add_argument('--train_batch_size', type=int, default=128, help='Training Batch Size')
        self.parser.add_argument('--test_batch_size', type=int, default=256, help='Testing Batch Size')
        self.parser.add_argument('--shuffle_buffer_size', type=int, default=1000, help='Size of shuffle buffer')
        self.parser.add_argument('--fetch_buffer_size', type=int, default=2, help='Size of shuffle buffer')
        self.parser.add_argument('--resample', type=int, default=0,
                                 help='Downsample the majority dataset and upsample the minority dataset')
        self.parser.add_argument('--weighted', type=int, default=0,
                                 help='Use weighted cross entropy loss')
        self.parser.add_argument('--model_type', type=str, help='Train_Type')
        self.parser.add_argument('--save_dir', type=str, default='shl_ckpt',help='Directory for saving model')
        self.parser.add_argument('--use_perception_loss', type=int, default=0, help='Use perception loss')
        self.parser.add_argument('--classifier_path', type=str, default='',
                                 help='Path to classification model for task based loss')
        self.parser.add_argument('--init_epochs',type=int,default=0,help='Initial epochs without adversarial loss')
        self.parser.add_argument('--model_path', type=str, default='',
                                 help='Path to generative model for evaluation results')
        self.parser.add_argument('--evaluate', type=int, default=0, help='Evaluate model on test dataset')

        # ===============================================================
        #                     GAN options
        # ===============================================================
        self.parser.add_argument('--gan_type', type=str,default='normal',help='Type of GAN Loss', choices=['normal','wgan','wgangp','normalls'])
        self.parser.add_argument('--clip_value',type=float,default=0.01,help='Clip value for WGAN')
        self.parser.add_argument('--gp_lambda',type=float,default=10,help='Multiplier for Gradient Penalty')

        # ===============================================================
        #                     Super Resolution Options
        # ===============================================================
        self.parser.add_argument('--use_sr_clf', type=int, default=0, help='Use super-resolution data for classification')
        self.parser.add_argument('--sampling_ratio', type=int, default=1, help='Downsampling factor')
        self.parser.add_argument('--interp', type=int, default=0, help='Use interpolation for upsampling' )
        self.parser.add_argument('--interp_type', type=str, default='linear', help='Default method of interpolation')

        # ===============================================================
        #                 Missing Data Imputation Options
        # ===============================================================
        self.parser.add_argument('-seed', type=int, default=1, help='Random Seed for missing data')
        self.parser.add_argument('--prob', type=float, default=0.0, help='Probability of missing data')
        self.parser.add_argument('--masked_mse_loss', type=int, default=1, help='Use MSE loss between imputed values')
        self.parser.add_argument('--cont', type=int, default=1, help='Continuous chunk missing')
        self.parser.add_argument('--use_imp_clf', type=int, default=0, help='Use missing_data_imputation for classification')
        self.parser.add_argument('--fixed', type=int, default=0, help='Fix the training dataset for imputation')



    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt