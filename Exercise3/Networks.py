import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
	def __init__(self, input_d, output_d):
		super(ValueNetwork, self).__init__()

		H = 25
		self.model = torch.nn.Sequential(
          torch.nn.Linear(input_d, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, output_d),
        )

	def forward(self, inputs):		
		return self.model(inputs)
