from Networks import ValueNetwork

input_d = 72
actions_n = 4

def create_network():
	return ValueNetwork(input_d=input_d, output_d=1)