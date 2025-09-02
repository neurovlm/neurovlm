"""Training loop."""

import warnings
from copy import deepcopy
from typing import Callable, Optional
import torch
from torch import nn
import numpy as np
from .progress import select_tqdm


class Trainer:
    """Training loop."""
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        lr: float,
        batch_size: int,
        n_epochs: int,
        optimizer: Callable,
        X_val: Optional[torch.tensor]=None,
        y_val: Optional[torch.tensor]=None,
        verbose: Optional[bool]=True,
        interval: Optional[int]=None,
        use_tqdm: Optional[bool]=False,
        tensorboard_path: Optional[str]=None,
        device: Optional[str]=None
    ):
        """Initialize training parameters.

        Parameters
        ----------
        model : torch.nn.Module
            Model to fit.
        loss_fn : Callable
            Loss function, e.g. torch.nn.MSELoss.
        lr : float
            Learning rate or step size.
        batch_size : int
            Size of mini-batches.
        n_epochs : int
            Number of epochs or full dataset passes through the model.
        optimizer : Callable
            Un-inialized torch optimizer. Use partial to set optional
            kwargs if needed.
        X_val : 2d torch.tensor, optional, default: None
            Validation input data.
        y_val : 2d torch.tensor, optional, default: None
            Validation target data.
        verbose : bool, optional, default: True
            Prints val loss after every epoch if True. Must pass X_val.
            Int
        interval : optional, default: None
            How often to traci val loss, in epochs.
        use_tqdm : bool, optional, default: False
            Training progess bar.
        tensorboard_path : bool, optional, default: False
            Path to store interactive webpage that displays validation loss over epochs.
        device : {None, "cuda", "mps", "cpu", "auto"}
            Moves model and tensors to requested device.
                - None: leaves leaves on current device
                - "auto": moves to gpu based on availablity
                - "mps": Apple
                - "cuda": Nivida
        """
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer(self.model.parameters(), self.lr)
        self.verbose = verbose
        self.interval = 1 if self.verbose and interval is None else interval
        self.iter_wrapper = select_tqdm() if use_tqdm else None
        self.X_val = X_val
        self.y_val = y_val


        if self.verbose and self.X_val is None:
            warnings.warn("No validation set to report on. Setting verbose to False.")
            self.verbose = False

        self.device = device
        if self.device == "auto":
            self.device = which_device()

        if self.device is not None:
            if self.X_val is not None:
                self.X_val = self.X_val.to(self.device)
            if self.y_val is not None:
                self.y_val = self.y_val.to(self.device)
            self.model = self.model.to(self.device)

        if self.y_val is None and self.X_val is not None:
            # Assume autoencoder, e.g. target is self
            self.y_val  = self.X_val

        self.tensorboard_path = tensorboard_path
        if self.tensorboard_path:
            from torch.utils.tensorboard import SummaryWriter
            import datetime
            log_dir = self.tensorboard_path + "/loss_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(log_dir)

    def fit(
        self,
        X_train: torch.tensor,
        y_train: Optional[torch.tensor]=None
    ):
        """Training loop.

        Parameters
        ----------
        X_train : 2d torch.tensor
            Training input data.
        y_train : 2d torch.tensor, optional, default: None
            Training target data. None assumes an autoencoder.
        """
        # Set device
        if self.device is not None:
            X_train = X_train.to(self.device)
            if y_train is not None:
                y_train = y_train.to(self.device)

        if y_train is None:
            # Autoencoder model
            y_train = X_train

        # Initial validation loss
        if self.verbose:
            with torch.no_grad():
                y_pred = self.model(self.X_val)
                init_loss = float(self.loss_fn(
                    y_pred, self.y_val
                ))
                print(f"Epoch: -1, val loss: {float(init_loss):.5g}")

        # Train text aligner
        if self.iter_wrapper is not None:
            iterable = self.iter_wrapper(range(self.n_epochs), total=self.n_epochs)
        else:
            iterable = range(self.n_epochs)

        self._best_model = None
        best_loss = np.inf

        for iepoch in iterable:
            # Randomly shuffle data
            torch.manual_seed(iepoch)
            rand_inds = torch.randperm(len(X_train))
            for i in range(0, len(X_train), self.batch_size):
                # Forward
                y_pred = self.model(
                    X_train[rand_inds[i:i+self.batch_size]]
                )
                # Backward
                loss = self.loss_fn(
                    y_pred,
                    y_train[rand_inds[i:i+self.batch_size]]
                )
                loss.backward()
                # Step
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (self.interval is not None and iepoch % self.interval == 0) or self.tensorboard_path is not None:

                # Report validation loss
                with torch.no_grad():

                    y_pred = self.model(self.X_val)
                    current_loss = self.loss_fn(y_pred, self.y_val)

                    if self.verbose:
                        print(f"Epoch: {iepoch}, val loss: {float(current_loss):.5g}")

                    if current_loss < best_loss:
                        best_loss = float(current_loss)
                        self._best_model = deepcopy(self.model)

                    if self.tensorboard_path is not None:
                        self.writer.add_scalar("Loss val", current_loss.item(), iepoch)

    def predict(self, X):
        """Foward pass.

        Parameters
        ----------
        X : 2d torch.tensor
            Dataset to evaluate.
        """
        return self.model(X)

    def save(self, path):
        """Save model.

        Parameters
        ----------
        path : str
            Where to save model to, including model name.
        """
        torch.save(self.model, path)

    def restore_best(self):
        """Restore the best model, basd on best validation loss."""
        self.model = self._best_model


def which_device() -> str:
    """Determine the device to move models and tensors to.

    Returns
    -------
    device : {"cuda", "mps", "cpu"}
    """
    # Set device
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device