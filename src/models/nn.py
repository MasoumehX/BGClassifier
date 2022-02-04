"""
@author: Masoumeh M. Trai
@place: University of Tuebingen
@date: January 2022

A class of SVM, CNN and VAE or LSTM
"""

import torch
import numpy as np
from tqdm import trange
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Linear, Module, ReLU, Tanh, NLLLoss, CrossEntropyLoss
import torch.nn as nn
from data import *


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=ReLU):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.activation = activation
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity  # NON-LINEAR
        out = self.activation(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out


def training(model: FeedforwardNeuralNetModel, X_train: Tensor, y_train: Tensor, X_test: Tensor, y_test: Tensor,
                     criterion: Module = CrossEntropyLoss(), steps=10000, p_norm: int = 2, norm_scale: float = 0.05,
                     lr: float = 0.1):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    running_losses = []
    iter = 0
    for epoch in range(steps):

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # model parameters
        model_params = torch.nn.utils.parameters_to_vector(model.parameters())

        # Forward pass to get output/logits
        outputs = model(X_train)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, y_train)

        # regularization
        if p_norm:
            if p_norm == 1:
                loss += norm_scale * model_params.norm(p_norm)
            elif p_norm == 2:
                loss += norm_scale * (model_params ** 2).sum()
            else:
                raise ValueError("Unknown norm, use '1' or '2'")

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1
        running_losses.append(loss.detach().numpy())
        if iter % 100 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset

            # Forward pass only to get logits/output
            outputs = model(X_test)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += y_test.size(0)

            # Total correct predictions
            correct += (predicted == y_test).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
    return np.array(running_losses)


if __name__ == '__main__':

    # read the data and create tensor
    where_to_read = "/home/masoumeh/Desktop/MasterThesis/Data/"
    filename = "dataset_small_small.csv"

    # server
    # txt_file_path = "/data/home/masoumeh/Data/classessem.txt"
    # csv_file_path = "/data/home/agora/data/rawData/fullColectionCSV/fullColectionCSV|2022-01-19|03:42:12.csv"

    X_data, y_data = create_data_for_nn(where_to_read+filename)
    X_train, X_test, y_train, y_test = split_train_test(X_data, y_data)

    # Tensor
    X_train = torch.Tensor(X_train).float()
    y_train = torch.Tensor(y_train).float()
    X_test = torch.Tensor(X_test).float()
    y_test = torch.Tensor(y_test).float()

    input_dim = len(X_data[1])
    hidden_dim = 100
    output_dim = len(set(y_data))

    print(input_dim, hidden_dim, output_dim)

    ff_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

    training(model=ff_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
