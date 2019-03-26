#!/usr/bin/env python3
# encoding utf-8


import os
import worker_factory
from logger import Logger
import torch
import argparse

os.environ['OMP_NUM_THREADS'] = '1'

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=15000)
	parser.add_argument('--experiment', type=str, default="exp_test")
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--n_workers', type=int, default=6)

	args = parser.parse_args()
	experiment = args.experiment

	logger = Logger("output_{0}.out".format(experiment))	
	
	logger.log("Starting experiment: {0}".format(experiment))
	logger.log(str(args))

	os.system("mkdir model_weights/{0}".format(experiment))

	config = {
		"n_workers": args.n_workers,
		"startingEpsilons": [1, 0.8, 0.5, 0.5, 0.3, 0.1, 0.7, 0.4],
		"minEpsilons": [0.4, 0.35, 0.25, 0.25, 0.15, 0, 0.3, 0],
		"numPolicyUpdates": 8000,
		"discountFactor": 0.99,			# worth exploring
		"learning_rate": args.lr,
		"learning_network_update_interval": 6,
		"target_network_update_interval": 200,
		"parameterStoragePath": "model_weights/{0}/weights_".format(experiment),
		"max_grads": 1,
		"parameter_save_frequency": 1000000,
		"numEpisodes": args.numEpisodes,
		"experiment": experiment
	}
	workers, target_network = worker_factory.create_workers(config, logger)
	logger.log("running {0} workers".format(len(workers)))
	for w in workers:
		w.join()

	logger.log("saving final network weights")
	torch.save(target_network.state_dict(), config["parameterStoragePath"] + "final.out")



