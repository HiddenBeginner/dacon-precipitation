import os
import pandas as pd

from sklearn.model_selection import KFold


def prepare_metadata(dir_input='./input', dir_metadata='./input/metadata.csv'):
    '''
    Prepare metadata in csv format.

    parameters
    ----------
    dir_input: str, the path of folder where the "train" and the "test" folder are located.
    dir_metadata: str or None, the path of the csv file to be saved. If dir_metadata is None, the csv file is not saved.
    '''
    # Create metadata ==================== #
    train = {'id': os.listdir(f'{dir_input}/train'),
             'dataset': ['train'] * len(os.listdir(f'{dir_input}/train'))}
    train = pd.DataFrame(train)

    test = {'id': os.listdir(f'{dir_input}/test'),
            'dataset': ['test'] * len(os.listdir(f'{dir_input}/test'))}
    test = pd.DataFrame(test)
    # End ================================ #
    # Split train data =================== #
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold_number, (train_index, val_index) in enumerate(kfold.split(range(train.shape[0]))):
        train.loc[val_index, 'dataset'] = f'fold_{str(fold_number)}'

    metadata = pd.concat([train, test], axis=0, ignore_index=True)
    # End ================================ #
    # Save metadata ###=================== #
    if dir_metadata is not None:
        metadata.to_csv(dir_metadata, index=False)
    # End ================================ #

    return metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_input', type=str, default='./input')
    parser.add_argument('--dir_metadata', type=str, default='./input/metadata.csv')
    args = parser.parse_args()

    prepare_metadata(args.dir_input, args.dir_metadata)
