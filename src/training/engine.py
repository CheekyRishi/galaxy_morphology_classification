"""
Contains functions for training and testing a PyTorch model.
"""
from typing import Dict, List, Tuple

import torch

from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() * y.size(0)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item()


    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader.dataset)
    train_acc = train_acc / len(dataloader.dataset)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item() * y.size(0)

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item()

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader.dataset)
    test_acc = test_acc / len(dataloader.dataset)

    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          validation: bool,
          early_stopping=None) -> Tuple[Dict[str, List[float]], int]:
    """
    Trains and evaluates a PyTorch model.

    The model is trained using `train_step()` and evaluated using
    `test_step()` for a specified number of epochs. If an
    EarlyStopping object is provided, training may stop before
    reaching the maximum number of epochs.

    Metrics are logged for each epoch, and the epoch corresponding
    to the best validation loss is tracked.

    Args:
        model: A PyTorch model to be trained and evaluated.
        train_dataloader: DataLoader for the training dataset.
        test_dataloader: DataLoader for the validation or test dataset.
        optimizer: Optimizer used to update model parameters.
        loss_fn: Loss function used for optimization.
        epochs: Maximum number of epochs to train for.
        device: Target device for computation ("cuda" or "cpu").
        validation: If True, metrics are logged as validation metrics.
        early_stopping: Optional EarlyStopping object to control
            early termination of training.

    Returns:
        A tuple containing:
            - results (Dict[str, List[float]]):
                Dictionary with per-epoch metrics.
                Keys include:
                    "train_loss", "train_acc",
                    "val_loss" / "test_loss",
                    "val_acc" / "test_acc"

            - best_epoch (Optional[int]):
                Epoch number corresponding to the lowest validation loss.
                Returns None if early stopping is not used.
    """
    
    eval_loss_key = "val_loss" if validation else "test_loss"
    eval_acc_key  = "val_acc"  if validation else "test_acc"

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               eval_loss_key: [],
               eval_acc_key: []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
       
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"{eval_loss_key}: {test_loss:.4f} | "
          f"{eval_acc_key}: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results[eval_loss_key].append(test_loss)
        results[eval_acc_key].append(test_acc)

        if early_stopping is not None:
            early_stopping.step(test_loss, epoch + 1)

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    best_epoch = early_stopping.best_epoch if early_stopping is not None else None

    return results, best_epoch


def train_vit(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    early_stopping,
    validation: bool = True,
    checkpoint_path: str = "vit_best.pth",
    verbose: bool = True
) -> Tuple[Dict[str, List[float]], int]:
    """
    Trains a Vision Transformer model.

    If validation=True:
        Uses validation dataloader and logs val_* metrics

    If validation=False:
        Uses test dataloader and logs test_* metrics
        (useful for balanced-dataset experiments)

    Returns:
        results: Dict with per-epoch metrics
        best_epoch: Epoch with lowest eval loss
    """

    eval_loss_key = "val_loss" if validation else "test_loss"
    eval_acc_key  = "val_acc"  if validation else "test_acc"

    results = {
        "train_loss": [],
        "train_acc": [],
        eval_loss_key: [],
        eval_acc_key: []
    }

    best_eval_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Training ViT"):
      
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        eval_loss, eval_acc = test_step(
            model=model,
            dataloader=eval_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results[eval_loss_key].append(eval_loss)
        results[eval_acc_key].append(eval_acc)

        if verbose:
            print(
                f"Epoch {epoch+1:03d} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"{eval_loss_key}: {eval_loss:.4f} | "
                f"{eval_acc_key}: {eval_acc:.4f}"
            )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), checkpoint_path)

        early_stopping.step(eval_loss, epoch + 1)
        if early_stopping.should_stop:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best epoch was {early_stopping.best_epoch}"
            )
            break

    return results, early_stopping.best_epoch
