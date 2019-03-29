from Networks import ValueNetwork

#input_d = 72    # for Low Level Features with action
# input_d = 19	# for High Level Features with action 
# input_d = 15	# for High Level Features without action 
input_d = 68    # for Low Level Features with action
actions_n = 4

def create_network(num_layers=2, hidden_size=25):
	network = ValueNetwork(input_d=input_d, output_d=actions_n, hidden_size=hidden_size)
	network.share_memory()
	return network
