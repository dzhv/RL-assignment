import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
	def __init__(self, input_d=68, output_d=4, num_layers=2, hidden_size=25):
		super(ValueNetwork, self).__init__()

		H = hidden_size

		if num_layers == 3:
			self.model = torch.nn.Sequential(
			  torch.nn.Linear(input_d, H),
			  torch.nn.ReLU(),
			  torch.nn.Linear(H, H),
			  torch.nn.ReLU(),
			  torch.nn.Linear(H, H),
			  torch.nn.ReLU(),
			  torch.nn.Linear(H, output_d),
			)
		else:
			self.model = torch.nn.Sequential(
			  torch.nn.Linear(input_d, H),
			  torch.nn.ReLU(),
			  torch.nn.Linear(H, H),
			  torch.nn.ReLU(),
			  torch.nn.Linear(H, output_d),
			)

	def forward(self, inputs):		
		return self.model(inputs)
