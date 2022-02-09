"""
@author: Masoumeh M. Trai
@place: University of Tuebingen
@date: January 2022

A class of SVM, CNN and VAE or LSTM
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, ReLU, Tanh, NLLLoss, CrossEntropyLoss
from data import *


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3 (readout): 100 --> 10
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3 (readout)
        out = self.fc3(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


def training(model, X_train: Tensor, y_train: Tensor, X_test: Tensor, y_test: Tensor,
                     criterion: Module = CrossEntropyLoss(), steps=1000, p_norm: int = 2, norm_scale: float = 0.05,
                     lr: float = 0.01):
    print("criterion", criterion)
    print("steps", steps)
    print("learning rate: ", lr)
    print("p norm: ", p_norm)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    running_losses = []
    iter = 0
    for epoch in range(steps):
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # model parameters
        model_params = torch.nn.utils.parameters_to_vector(model.parameters())
        # Forward pass to get output/logits
        X_train = X_train.requires_grad_()
        outputs = model(X_train)
        # Calculate Loss: softmax --> cross entropy loss
        y_train = y_train.long()
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
        # iter += 1
        running_losses.append(loss.detach().numpy())
        # Calculate Accuracy
        correct = 0
        total = 0
        # Iterate through test dataset
        # Forward pass only to get logits/output
        X_test = X_test.requires_grad_()
        outputs = model(X_test)
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        y_test = y_test.long()
        total += y_test.size(0)
        # Total correct predictions
        correct += (predicted == y_test).sum()
        accuracy = 100 * correct / total
        # Print Loss
        print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
        # if iter % 100 == 0:
    return np.array(running_losses)


if __name__ == '__main__':

    # read the data and create tensor
    # where_to_read = "/home/masoumeh/Desktop/MasterThesis/Data/"
    # filename = "dataset_small_small.csv"

    # server
    where_to_read = "/data/home/masoumeh/Data/"
    train = "train_big_clean.csv"
    test = "test_big_clean.csv"

    df_train = read_csv_file(where_to_read+train)
    df_test = read_csv_file(where_to_read+test)
    X_train, y_train = create_data_for_nn(df_train)
    X_data, y_data = create_data_for_nn(df_train)
    # X_train, X_test, y_train, y_test = split_train_test(X_data, y_data)

    features = ["x", "y", "poi"]
    X_train = df_train[features].values
    y_train = df_train["classes"].values

    X_test = df_test[features].values
    y_test = df_test["classes"].values

    # Tensor
    X_train = torch.Tensor(X_train).float()
    y_train = torch.Tensor(y_train).long()
    X_test = torch.Tensor(X_test).float()
    y_test = torch.Tensor(y_test).long()

    input_dim = 2
    hidden_dim = 100
    layer_dim = 3
    output_dim = 3

    print(input_dim, hidden_dim, output_dim)

    # FFN
    # model = FeedforwardNeuralNetModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # LSTM
    LSTM_model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    training(LSTM_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, lr=0.1)
