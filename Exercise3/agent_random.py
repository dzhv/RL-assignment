#!/usr/bin/env python3
# encoding utf-8

import os
import policy_worker 
from Environment import HFOEnv
import network_factory
import torch
from Policy import RandomPolicy

os.environ['OMP_NUM_THREADS'] = '1'

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :
	
	rnd_seed = 11111 + 111
	environment = HFOEnv(port=6011, seed=rnd_seed, numOpponents=1)
	environment.connectToServer()

	policy_worker.run(num_episodes=1000, value_network={},
		environment=environment, policy=RandomPolicy())

	environment.quitGame()

