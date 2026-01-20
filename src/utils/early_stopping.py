import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.best_epoch = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss, epoch):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


