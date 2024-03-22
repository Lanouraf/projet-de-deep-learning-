"""Functions for the deep learning mode.

This script contains functions for training and evaluating a deep learning model. It includes a training loop, a function to compute accuracy, and other utility functions.

The main function in this script is `fit`, which performs the training loop for a given model. It takes the model, training data, validation data, number of epochs, loss function, learning rate, and other parameters as inputs. It trains the model using the provided data and returns the training losses, validation losses, and accuracy for each epoch.

Other functions in this script include:
- `accuracy`: Computes the accuracy of the network given the output predictions and ground truth labels.

Note: This script requires the following dependencies: torch, torch.nn, torch.optim, tqdm, matplotlib.pyplot.

"""


import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import streamlit as st

    ########################################
    
    
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt


def fit(model, train_dataloader, valid_dataloader, epochs, loss_fn=nn.CrossEntropyLoss(), lr=1e-3, plot_each=20):
    """
    Simple training loop for a given model
    :param model: Model to be trained
    :param train_dataloader: Training data
    :param valid_dataloader: Validation data
    :param epochs: Number of epochs
    :param loss_fn: Loss function
    :param lr: Learning rate
    :param plot_each: Output progress each "plot_each" epochs
    :return: training losses (per mini-batch and epoch), validation losses (per epoch), accuracy (per epoch)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr = lr)
    valid_loss = 0  # Last epoch validation loss for display
    valid_losses = []
    losses = []
    last_accuracy = 0
    accuracies = []

    progbar = tqdm(range(epochs))
    for epoch in progbar:
        # Train one epoch
        losses_batch = []
        model.train()
        for i, batch in enumerate(train_dataloader):
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            losses_batch.append(loss.item())
            if i % plot_each == 0:
                progbar.set_postfix_str(
                    f'Batch: {i}/{len(train_dataloader)}  Loss: {loss.item()} Validation: {valid_loss} Accuracy: {last_accuracy}')
        losses.append(losses_batch)

        # Validation
        model.eval()
        accuracy_batch = 0
        with torch.no_grad():
            valid_loss = 0
            for batch in valid_dataloader:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss_batch = loss_fn(pred, yb)
                valid_loss += loss_batch.item()

                accuracy_batch += accuracy(pred, yb).item()
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
        last_accuracy = accuracy_batch / len(valid_dataloader)
        accuracies.append(last_accuracy)

    return losses, valid_losses, accuracies


def accuracy(preds, y):
    """
    Computes the accuracy of the network
    :param preds: Output predictions from the model
    :param y: Categorical labels (Ground truth)
    :return: Accuracy
    """
    preds = torch.argmax(preds, dim = 1)
    return sum(preds == y) / len(preds)
