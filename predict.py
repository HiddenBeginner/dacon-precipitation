from unet import UNet
from fitter import Fitter
from datasets import PrecipitationDataset

import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader


def get_arguments():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument('--dir_meta', type=str, default='./input/metadata.csv', help='The path of metadata')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    # Fitter arguments
    parser.add_argument('--dir_checkpoint', type=str,
                        default='./output/checkpoints/last-checkpoint.bin',
                        help='The checkpoint path')
    parser.add_argument('--dir_sub', type=str, default='./output/sub/submission.csv',
                        help='The path where trained model will be saved')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    df = pd.read_csv(args.dir_meta)

    test_dataset = PrecipitationDataset(df=df[df['dataset'] == 'test'].reset_index(drop=True),
                                        dir_sample='./input/test',
                                        mode='test',
                                        imsize=120,
                                        transform=None)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False)

    model = UNet(n_channels=4, n_classes=1)
    fitter = Fitter(model,
                    verbose=False)
    fitter.load(args.dir_checkpoint)
    results = fitter.predict(test_loader)

    filename = pd.DataFrame(results['file_name'], columns=['file_name'])
    pixels = pd.DataFrame(np.array(results['pixels'], dtype=np.uint8), columns=[i for i in range(14400)])

    sub = pd.concat([filename, pixels], axis=1).to_csv(args.dir_sub, index=False)
