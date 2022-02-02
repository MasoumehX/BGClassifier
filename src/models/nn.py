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
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out


def learninglearning(model: FeedforwardNeuralNetModel, train: Tensor, test: Tensor, criterion: CrossEntropyLoss(),
                     steps=10000, p_norm: int = 2, norm_scale: float = 0.05, lr: float = 0.1, weight_decay=0):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    iter = 0
    for epoch in range(steps):
        for i, (images, labels) in enumerate(train):

            # Load images with gradient accumulation capabilities
            images = images.view(-1, 28 * 28).requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # model parameters
            model_params = torch.nn.utils.parameters_to_vector(model.parameters())

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

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

            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test:
                    # Load images with gradient accumulation capabilities
                    images = images.view(-1, 28 * 28).requires_grad_()

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / total

                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


# class FeedForward(torch.nn.Module):
#     def __init__(self, n_features: int, n_classes: int, hidden_size: int = 10, activation=ReLU, init=False):
#         """Construct a feed-forward neural network with the given number of
#            input features, hidden size, class size, and activation function (non-linear
#            function)."""
#         super(FeedForward, self).__init__()
#         self.hid1 = Linear(n_features, hidden_size)
#         self.hid2 = Linear(hidden_size, hidden_size)
#         self.non_linear = activation()
#         self.output = Linear(hidden_size, n_classes)
#
#         # initialization the weights and bias
#         if init:
#             torch.nn.init.xavier_uniform_(self.hid1.weight)
#             torch.nn.init.zeros_(self.hid1.bias)
#             torch.nn.init.xavier_uniform_(self.hid2.weight)
#             torch.nn.init.zeros_(self.hid2.bias)
#             torch.nn.init.xavier_uniform_(self.output.weight)
#             torch.nn.init.zeros_(self.output.bias)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Compute the raw activations (wx+b) for all instances.
#         Given an `x` of shape `[n_instances, n_features]`, returns activations with shape
#         `[n_instances]`.
#         """
#         z = self.hid1(x)
#         z = self.hid2(z)
#         act = self.non_linear(self.hid2(z))
#         out = self.output(act)
#         softmax = torch.nn.Softmax(dim=1)
#         return torch.argmax(softmax(out)).squeeze()
#
#
# def learning(model: FeedForward, x: Tensor, y: Tensor, loss_function: Module = CrossEntropyLoss(), steps=10000,
#              p_norm: int = 2, norm_scale: float = 0.05, lr: float = 0.1, weight_decay=0):
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
#     norms = []
#     running_losses = []
#
#     for step in trange(steps, desc="optimizing model"):
#         optimizer.zero_grad()
#         model_params = torch.nn.utils.parameters_to_vector(model.parameters())
#         loss = loss_function(model(x), y.type(torch.long))
#         if p_norm:
#             if p_norm == 1:
#                 loss += norm_scale * model_params.norm(p_norm)
#             elif p_norm == 2:
#                 loss += norm_scale * (model_params ** 2).sum()
#             else:
#                 raise ValueError("Unknown norm, use '1' or '2'")
#         loss.backward()
#         optimizer.step()
#         norms.append(float(model_params.norm()))
#         running_losses.append(loss.detach().numpy())
#         with torch.no_grad():
#             acc = model(x).eq(y).to(torch.float).mean()
#             print(f"Step: {step}, loss: {loss}, acc: {acc}")
#     return np.array(running_losses)



