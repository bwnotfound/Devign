import numpy as np


# Author: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, model, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False
        self.val_acc_min = 0
        self.delta = delta
        self.model = model

    def __call__(self, val_acc):

        if val_acc < self.best_acc + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_acc = val_acc
            self.save_checkpoint(val_acc)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_acc):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation acc incresed ({self.val_acc_min:.4f} --> {val_acc:.4f}).  Saving model ...')
        self.model.save()
        self.val_acc_min = val_acc
