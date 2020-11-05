import numpy as np

import torch
from torch.utils.data import Dataset


class PrecipitationDataset(Dataset):
    def __init__(self, df, dir_sample, imsize, mode, transform=None, p_cutmix=0.0, p_mixup=0.0):
        self.df = df
        self.dir_sample = dir_sample
        self.imsize = imsize
        self.mode = mode
        self.transform = transform
        self.p_cutmix = p_cutmix
        self.p_mixup = p_mixup

        if self.mode not in ['train', 'test']:
            raise AttributeError('The "mode" must be one of ["train", "test"]')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        sample_id = self.df.loc[index, 'id']

        if self.mode == 'test':
            X, y = self.load_sample_and_label(index)

        else:
            if self.p_cutmix + self.p_mixup > 1:
                raise AttributeError("The sum of probabilities of cutmix and mixup have to be less than or equal to 1.0")
            else:
                sample_type = np.random.choice(a=[0, 1, 2],
                                               p=[1 - (self.p_cutmix + self.p_mixup), self.p_cutmix, self.p_mixup])
                if sample_type == 0:
                    X, y = self.load_sample_and_label(index)
                elif sample_type == 1:
                    X, y = self.load_cutmix_sample_and_label(index, imsize=self.imsize)
                else:
                    X, y = self.load_mixup_sample_and_label(index)

        X = torch.as_tensor(X).permute(2, 0, 1)
        y = torch.as_tensor(y).permute(2, 0, 1)

        return X, y, sample_id

    def load_sample_and_label(self, index):
        sample_id = self.df.loc[index, 'id']
        sample = np.load(f'{self.dir_sample}/{sample_id}').astype(np.float32)
        if self.mode == 'test':
            sample = np.concatenate((sample, np.zeros((self.imsize, self.imsize, 1), dtype=np.float32)), axis=-1)

        sample = sample / 255.

        X = sample[:, :, :4]
        y = sample[:, :, 4:5]

        return X, y

    def load_cutmix_sample_and_label(self, index, imsize):
        """
        This implementation of cutmix author: https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(np.random.uniform(imsize*0.25, imsize*0.75)) for _ in range(2)]
        # Choose three random images
        indices = [index] + [np.random.randint(0, self.__len__() - 1) for _ in range(3)]

        # Initialize the result image
        result_X = np.full((w, h, 4), 1, dtype=np.float32)
        result_y = np.full((w, h, 1), 1, dtype=np.float32)

        for i, index in enumerate(indices):
            X, y = self.load_sample_and_label(index)

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # top left
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc  # top right
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)  # bottom left
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)  # bottom right

            result_X[y1a:y2a, x1a:x2a] = X[y1a:y2a, x1a:x2a]
            result_y[y1a:y2a, x1a:x2a] = y[y1a:y2a, x1a:x2a]

        return result_X, result_y

    def load_mixup_sample_and_label(self, index):
        target_index = np.random.randint(0, self.__len__()-1)
        alpha = np.clip(np.random.beta(1.0, 1.0), 0.2, 0.8)

        X1, y1 = self.load_sample_and_label(index)
        X2, y2 = self.load_sample_and_label(target_index)

        X = alpha * X1 + (1 - alpha) * X2
        y = alpha * y1 + (1 - alpha) * y2

        return X, y
