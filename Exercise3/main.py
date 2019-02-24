#!/usr/bin/env python3
# encoding utf-8


import os
import worker_factory

os.environ['OMP_NUM_THREADS'] = '1'

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :

	config = {
		"n_workers": 1,
		"epsilons": [1, 0.5, 0.75],
		"discountFactor": 0.99,
		"learning_rate": 0.1,
		"learning_network_update_interval": 6,
		"target_network_update_interval": 200,
		"parameterStoragePath": "model_weights/weights_",
		"parameter_save_frequency": 1000000,
		"numEpisodes": 1000,
	}
	workers = worker_factory.create_workers(config)
	for w in workers:
		w.join()



