"""
@author: Masoumeh M. Trai
@place: University of Tuebingen
@date: January 2022

A class of SVM, CNN and VAE or LSTM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class SVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 400)

    # def forward(self, x):

