from unet import UNet
from fitter import Fitter
from transforms import get_transform
from datasets import PrecipitationDataset

import argparse
import pandas as pd

from torch.utils.data import DataLoader


def get_arguments():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument('--dir_meta', type=str, default='./input/metadata.csv', help='The path of metadata')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--p_cutmix', type=float, default=0.0, help='Probability of cutmix')
    parser.add_argument('--p_mixup', type=float, default=0.0, help='Probability of mixup')

    # Fitter arguments
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of epochs')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether or not to print train/eval progression')
    parser.add_argument('--dir_base', type=str, default='./output/checkpoints',
                        help='The path where trained model will be saved')
    parser.add_argument('--dir_checkpoint', type=str, default=None,
                        help='The path of a pretrained model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    df = pd.read_csv(args.dir_meta)

    train_dataset = PrecipitationDataset(df=df[df['dataset'].isin(['fold_1', 'fold_2', 'fold_3', 'fold_4'])].reset_index(drop=True).copy(),
                                         dir_sample='./input/train',
                                         mode='train',
                                         imsize=120,
                                         p_cutmix=args.p_cutmix,
                                         p_mixup=args.p_mixup,
                                         transform=get_transform('train'))

    valid_dataset = PrecipitationDataset(df=df[df['dataset'] == 'fold_0'].reset_index(drop=True).copy(),
                                         dir_sample='./input/train',
                                         mode='test',
                                         imsize=120,
                                         transform=get_transform('test'))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False)

    model = UNet(n_channels=4, n_classes=1)

    fitter = Fitter(model,
                    lr=args.lr,
                    n_epochs=args.n_epochs,
                    verbose=args.verbose,
                    dir_base=args.dir_base)

    if args.dir_checkpoint is not None:
        fitter.load(args.dir_checkpoint)
        print(f"Pretrained weights are equipped. Epoch : {fitter.epoch}")

    fitter.fit(train_loader, valid_loader)

