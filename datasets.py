import numpy as np
from torch.utils.data import Dataset


class PrecipitationDataset(Dataset):
    class PrecipitationDataset(Dataset):
        def __init__(self, df, dir_sample, resize, mode, transform):
            self.df = df
            self.dir_sample = dir_sample
            self.resize = resize
            self.mode = mode
            self.transform = transform

            if self.mode not in ['train', 'test']:
                raise AttributeError('The "mode" must be one of ["train", "test"]')

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, index):
            sample_id = self.df.loc[index, 'id']

            X, y = self.load_sample_and_label(sample_id)

            return X, y

        def load_sample_and_label(self, sample_id):
            sample = np.load(f'{self.dir_sample}/{sample_id}').astype(np.float32)
            if self.mode == 'test':
                sample = np.concatenate((sample, np.zeros((self.resize, self.resize, 1))), axis=-1)

            sample = sample / 255.

            X = sample[:, :, :4]
            y = sample[:, :, 4]

            return X, y

