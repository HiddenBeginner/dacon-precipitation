from unet import UNet
from fitter import Fitter
from datasets import PrecipitationDataset

import argparse
import pandas as pd

from torch.utils.data import DataLoader


def get_arguments():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument('--dir_meta', type=str, default='./input/metadata(1102).csv', help='The path of metadata')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    # Fitter arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='The number of epochs')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether or not to print train/eval progression')
    parser.add_argument('--dir_base', type=str, default='./output/checkpoints', help='The path where trained model will be saved')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    df = pd.read_csv(args.dir_meta)

    train_dataset = PrecipitationDataset(df=df[df['dataset'] == 'train'].reset_index(drop=True),
                                         dir_sample='./input/train',
                                         mode='train',
                                         resize=120,
                                         transform=None)

    valid_dataset = PrecipitationDataset(df=df[df['dataset'] == 'validation'].reset_index(drop=True),
                                         dir_sample='./input/train',
                                         mode='test',
                                         resize=120,
                                         transform=None)

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

    fitter.fit(train_loader, valid_loader)

