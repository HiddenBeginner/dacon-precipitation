import os
import pandas as pd

from sklearn.model_selection import train_test_split


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

    metadata = pd.concat([train, test], axis=0, ignore_index=True)
    # End ================================ #
    # Split train data =================== #
    train_index, val_index = train_test_split(
        range(len(os.listdir(f'{dir_input}/train'))),
        test_size=0.2,
        shuffle=True,
        random_state=1
    )
    metadata.loc[val_index, 'dataset'] = 'validation'
    # End ================================ #
    # Save metadata ###=================== #
    if dir_metadata is not None:
        metadata.to_csv(dir_metadata, index=False)
    # End ================================ #

    return metadata