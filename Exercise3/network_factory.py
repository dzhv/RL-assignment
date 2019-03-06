from Networks import ValueNetwork

#input_d = 72    # for Low Level Features
input_d = 19	# for High Level Features
actions_n = 4

def create_network():
	return ValueNetwork(input_d=input_d, output_d=1)