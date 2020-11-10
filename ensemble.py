from glob import glob

import argparse
import numpy as np
import pandas as pd


class AverageEnsembler:
    def __init__(self, dir_sample_submission, dir_subs, dir_result_sub):
        self.sample_submission = pd.read_csv(dir_sample_submission)
        self.list_subs = glob(f'{dir_subs}/*.csv')
        self.dir_result_sub = dir_result_sub

        self.shape = (2674, 14400)

        # Initialization
        self.is_valid()
        self.result = np.zeros(self.shape, dtype=np.float32)
        self.n = 0

    def ensemble(self):
        for dir_sub in self.list_subs:
            sub = pd.read_csv(dir_sub)
            self.update(sub)

        self.result /= self.n
        self.sample_submission.iloc[:, 1:] = self.result.astype(np.uint8)
        self.sample_submission.to_csv(self.dir_result_sub, index=False)

    def update(self, df):
        pred = df.iloc[:, 1:].values.astype(np.float32)

        self.result += pred
        self.n += 1

    def is_valid(self):
        print("Check the validation of submission files ... ")
        shape = self.sample_submission.shape
        file_name = self.sample_submission['file_name'].values

        for dir_sub in self.list_subs:
            sub = pd.read_csv(dir_sub)
            if np.any(shape != sub.shape):
                raise AttributeError(
                    f'Invalid submission shape. Expected shape is {shape}, but {dir_sub} has the shape {sub.shape}'
                )

            elif np.any(file_name != sub['file_name'].values):
                raise AttributeError(f'The order of "file_name" in {df} does not match.')

        print("Finish the validation check.")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_sample_submission', type=str, default='./output/sub/sample_submission.csv')
    parser.add_argument('--dir_subs', type=str, default='./output/sub/to_ensemble')
    parser.add_argument('--dir_result_sub', type=str, default='./output/sub/ensemble.csv')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    ensembler = AverageEnsembler(dir_sample_submission=args.dir_sample_submission,
                                 dir_subs=args.dir_subs,
                                 dir_result_sub=args.dir_result_sub)
    ensembler.ensemble()
