import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path

        self.best_loss = np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss, model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
