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

	experiment_name = "first"
	os.system("mkdir model_weights/{0}".format(experiment_name))

	config = {
		"n_workers": 5,
		"epsilons": [1, 0.75, 0.5, 0.25, 0],
		"discountFactor": 0.99,
		"learning_rate": 1e-4,
		"learning_network_update_interval": 6,
		"target_network_update_interval": 200,
		"parameterStoragePath": "model_weights/{0}/weights_".format(experiment_name),
		"parameter_save_frequency": 1000000,
		"numEpisodes": 8000,
	}
	workers = worker_factory.create_workers(config)
	for w in workers:
		w.join()



