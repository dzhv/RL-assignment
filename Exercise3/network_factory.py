from Networks import ValueNetwork

#input_d = 72    # for Low Level Features with action
# input_d = 19	# for High Level Features with action 
# input_d = 15	# for High Level Features without action 
input_d = 68    # for Low Level Features with action
actions_n = 4

def create_network():
	network = ValueNetwork(input_d=input_d, output_d=actions_n)
	network.share_memory()
	return network
