#!/usr/bin/env python3
# encoding utf-8


import os
import worker_factory
from logger import Logger
import torch

os.environ['OMP_NUM_THREADS'] = '1'

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :

	logger = Logger("output.out")

	experiment_name = "exp5"
	logger.log("Starting experiment: {0}".format(experiment_name))
	os.system("mkdir model_weights/{0}".format(experiment_name))

	config = {
		"n_workers": 6,
		"startingEpsilons": [1, 0.8, 0.5, 0.5, 0.3, 0.1],
		"minEpsilons": [0.5, 0.4, 0.25, 0.25, 0.15, 0],
		"numPolicyUpdates": 7000,
		"discountFactor": 0.97,
		"learning_rate": 5e-5,
		"learning_network_update_interval": 6,
		"target_network_update_interval": 200,
		"parameterStoragePath": "model_weights/{0}/weights_".format(experiment_name),
		"max_grads": 1,
		"parameter_save_frequency": 1000000,
		"numEpisodes": 10000,
	}
	workers, target_network = worker_factory.create_workers(config, logger)
	logger.log("running {0} workers".format(len(workers)))
	for w in workers:
		w.join()

	logger.log("saving final network weights")
	torch.save(target_network.state_dict(), config["parameterStoragePath"] + "final.out")



