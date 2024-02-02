import argparse, os
from os.path import join


def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='Deep Nueral Network')

# path and dirs
parser.add_argument('--logs_dir', type=str, default='./logs',
                    help='Dir in where logs and model checkpoints are stored')

# optimization params
parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='Initial learning rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of images in each batch of data')
parser.add_argument('--num_workers', type=int, default=16,
                    help='num of workers to use')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train for')
parser.add_argument('--shuffle', type=str2bool, default=True,
                    help='Whether to shuffle the dataset between epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value')
parser.add_argument('--lr_patience', type=int, default=1,
                    help='Number of epochs to wait before reducing lr')
parser.add_argument('--train_patience', type=int, default=20,
                    help='Number of epochs to wait before stopping train')
parser.add_argument('--weight_decay', type=float, default=1e-3, 
                    help='Weight decay (default: 1e-3)')
parser.add_argument('--optimizer', type=str, default='SGD', 
                    help='Adam | SGD | LARS | RMSProp')
parser.add_argument('--flush', type=str2bool, default=False,
                    help='Whether to delete ckpt + log files for model no')
parser.add_argument('--trial', type=int, default=1, required=True,
                    help='id for recording multiple runs')
parser.add_argument('--best', type=str2bool, default=True,
                    help='Load best model or most recent for testing')
parser.add_argument('--random_seed', type=int, default=41,
                    help='Seed to ensure reproducibility')
parser.add_argument('--resume', type=str2bool, default=False,
                    help='Whether to resume training from checkpoint')
parser.add_argument('--fold_num', type=int, default=10, 
                    help='fold number out within num_folds range')
parser.add_argument('--num_folds', type=int, default=10, 
                    help='Number of folds (between 5 and 10) in cross-validation')
parser.add_argument('--learning_stage', type=str, default='train',
                    help='Whether to train, or test the model or generate the features')
parser.add_argument('--early_stop', type=str2bool, default=True,
                    help='Early stops the training if validation loss/accuracy does not improve after a given patience.')
parser.add_argument('--early_stop_on_loss', type=str2bool, default=True,
                    help='Early stops the training if validation loss does not improve after a given patience.')

# model params
parser.add_argument('--folder_name', type=str,
                    help='(no_age_sex_harmonizedFNC | FzTransformedOrigFNC')
parser.add_argument('--num_classes', type=int, default=2, 
                    help='number of classes')
parser.add_argument('--output_size', type=int, default=1, 
                    help='size of the output layer')
parser.add_argument('--num_layers', type=int, default=1, 
                    help='Number of layers of LSTM')
parser.add_argument('--drop_prob', type=float, default=0.5, 
                    help='')
parser.add_argument('--model', type=str, default='fnn',
                    help='(fnn | BrainNetCNN | GCNN')
parser.add_argument('--num_ftrs', type=int, default=3, 
                    help='number of features in the layer before classifier')
parser.add_argument('--dropout', type=float, default=0.1, 
                    help='')
parser.add_argument('--classification_way', type=str, default='AD_SZ', 
                    help='classification way: AD_SZ | AD_SZ_NC | AD_NC | SZ_NC')



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


