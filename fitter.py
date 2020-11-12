from pytorch_ssim import SSIM

import os
import time
import torch

import numpy as np

from glob import glob
from datetime import datetime


class Fitter:
    def __init__(self,
                 model,
                 lr=1e-1,
                 n_epochs=10,
                 verbose=True,
                 dir_base='./output/checkpoints'):
        self.model = model
        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose

        # Initializations
        self.dir_base = dir_base
        if not os.path.exists(self.dir_base):
            os.makedirs(self.dir_base)
        self.dir_log = f'{dir_base}/log.txt'
        self.best_summary_loss = 10**5
        self.epoch = 0

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        # Define the optimizer
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=1,
                                                                    verbose=True,
                                                                    threshold=0.00005,
                                                                    threshold_mode='abs',
                                                                    cooldown=0,
                                                                    min_lr=1e-8,
                                                                    eps=1e-8)

        # Define the loss
        self.criterion = SSIM()
        self.log(f'====================================================')
        self.log(f'Fitter prepared | Time: {datetime.utcnow().isoformat()} | Device: {self.device}')

    def fit(self, train_loader, valid_loader):
        for epoch in range(self.n_epochs):
            if self.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp} | LR {lr}')

            # Train
            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)
            self.log(
                f'[RESULT]: Train | Epoch: {self.epoch} | ' +
                f'summary_loss: {summary_loss.avg:.7f} | ' +
                f'time: {time.time() - t:.3f}'
            )
            self.save(f'{self.dir_base}/last-checkpoint.bin')

            # Validation
            t = time.time()
            summary_loss = self.validation(valid_loader)
            self.log(
                f'[RESULT]: Validation | Epoch: {self.epoch} | ' +
                f'summary_loss: {summary_loss.avg:.7f} | ' +
                f'time: {time.time() - t:.3f}'
            )

            # Save the best model
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.dir_base}/best-checkpoint-{str(self.epoch).zfill(3)}.bin')
                # Keep top 3 models
                for path in sorted(glob(f'{self.dir_base}/best-checkpoint-*.bin'))[:-3]:
                    os.remove(path)

            # Scheduler update
            self.scheduler.step(metrics=summary_loss.avg)
            self.epoch += 1

        # End for (overall epochs)

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()

        for step, (X, y, _) in enumerate(train_loader):
            if self.verbose:
                print(
                    f'Train step: {step+1}/{len(train_loader)} | ' +
                    f'Summary_loss: {summary_loss.avg:.7f} | ' +
                    f'Time: {time.time() - t:.3f} |', end='\r'
                )

            X = X.to(self.device)
            batch_size = X.shape[0]
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(X)
            loss = -1 * self.criterion(output, y)
            loss.backward()
            summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()
        # End for (one epoch)
        return summary_loss

    def validation(self, valid_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (X, y, _) in enumerate(valid_loader):
            if self.verbose:
                print(
                    f'Validation Step: {step + 1}/{len(valid_loader)} | ' +
                    f'Summary_loss: {summary_loss.avg:.7f} | ' +
                    f'Time: {time.time() - t:.3f} |', end='\r'
                )

            with torch.no_grad():
                X = X.to(self.device)
                batch_size = X.shape[0]
                y = y.to(self.device)

                output = self.model(X)
                loss = -1 * self.criterion(output, y)
                summary_loss.update(loss.detach().item(), batch_size)
        # End for (one epoch)
        return summary_loss

    def predict(self, test_loader):
        self.model.eval()
        t = time.time()
        results = {'file_name':[],
                   'pixels': []}
        for step, (X, y, sample_id) in enumerate(test_loader):
            print(f'Step: {step + 1}/{len(test_loader)} | ' +
                  f'Time: {time.time() - t:.3f} ', end='\r')

            X = X.to(self.device)
            output = self.model(X)
            output = postprocess(output)

            for i in range(len(sample_id)):
                results['file_name'].append(sample_id[i])
                results['pixels'].append(output[i].flatten().tolist())
            # End for (one batch)
        # End for (over all test data)
        return results

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch + 1
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch']

    def log(self, message):
        if self.verbose:
            print(message)
        with open(self.dir_log, 'a+') as logger:
            logger.write(f'{message}\n')


def postprocess(output):
    """
    Post-process a batch of output. The postprocessing follows a number of steps:
     0. Unify the data type of output with numpy.ndarray with shape (batch_size, 120, 120, 1)
     1. Clip output from 0 to 255.
     2. Set pixel values less than 1/255 to zero.
     3. Multiply output by 255.

    parameters
    ----------
    output: numpy.ndarray or torch.tensor, a batch of output.
    """
    # Make the shape of output (batch_size, 120, 120, 1)
    if isinstance(output, torch.cuda.FloatTensor):
        output = output.detach().cpu().numpy()

    elif isinstance(output, torch.FloatTensor):
        output = output.numpy()

    if output.shape[-1] != 1:
        output = output.transpose(0, 2, 3, 1)

    # Vanish pixels whose values are less than 1/510
    output[np.where(output < 1 / 510.)] = 0

    # Clip and rescale output from 0 to 255
    output = np.clip(output, 0, 1)
    output = output * 255
    output = np.round(output)

    return output.astype(np.uint8)


class AverageMeter(object):
    '''
    Compute and store the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

