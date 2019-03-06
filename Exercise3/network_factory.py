from Networks import ValueNetwork

#input_d = 72    # for Low Level Features with action
# input_d = 19	# for High Level Features with action 
input_d = 15
actions_n = 4

def create_network():
	return ValueNetwork(input_d=input_d, output_d=actions_n)