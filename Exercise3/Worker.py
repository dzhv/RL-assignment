import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from Networks import ValueNetwork
from Environment import HFOEnv
from Policy import Policy
import random
from collections import deque
import numpy as np
from Environment import HFOEnv

import os

policy = Policy()

def create_environment(idx):
    port = 6000 + idx * 4
    rnd_seed = 11111 * idx + 111
    environment = HFOEnv(port=port, seed=rnd_seed, numOpponents=1)
    environment.connectToServer()
    return environment

def increment_counter(counter):
	with counter.get_lock():
		counter.value += 1
		return counter.value

def train(idx, networks, optimizer, counter, policy, config, logger):
	environment = create_environment(idx)

	counterValue = 0
	for i in range(config["numEpisodes"]):
		logger.log("Episode {0}/{1}, for process: {2}".format(i, config["numEpisodes"], idx))
		logger.log("Counter: {0}".format(counterValue))

		experienceQueue = deque([])
		# gal dar kažką reik resetint?
		# kokius gradient, optimizer ka nors
		environment.reset()
		policy.updateEpsilon()

		while True:
			counterValue = increment_counter(counter)

			state = environment.curState
			action, q_value = policy.egreedyAction(state, computePredictions(state, networks["learning"]))
			nextState, reward, done, status, info = environment.step(environment.possibleActions[action])
			experienceQueue.append((state, action, reward, nextState))

			if done or len(experienceQueue) >= config["learning_network_update_interval"]:
				loss = 0 
				for i in range(len(experienceQueue)):
					state, action, reward, nextState = experienceQueue.popleft()

					target = computeTargets(reward, nextState, config["discountFactor"], done, networks["target"])
					loss += (target - computePrediction(state, action, networks["learning"]))**2

				optimizer.zero_grad()
				loss.backward()

				nn.utils.clip_grad_norm_(networks["learning"].parameters(), max_norm=40)

				optimizer.step()			

			if counterValue % config["target_network_update_interval"] == 0:
				networks["target"].load_state_dict(networks["learning"].state_dict())

			if counterValue % config["parameter_save_frequency"] == 0:
				logger.log("Saving value network, counter: {0}".format(counterValue))
				saveModelNetwork(networks["learning"], 
					config["parameterStoragePath"] + str(counterValue // config["parameter_save_frequency"]) + ".out")

			if done:
				break	

def computePredictions(state, network):
	state = state.view(-1)
	return network(state)

def computePrediction(state, action, valueNetwork):
	# return valueNetwork(state)[action]
	return computePredictions(state, valueNetwork)[action]

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	# get q value of the greedy action
	if done:
		return torch.Tensor([reward])

	predictions = computePredictions(nextObservation, targetNetwork)
	_, qmax = policy.greedyAction(predictions)
	return reward + discountFactor * qmax
	
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)

def one_hot_encode(action):
	action_n = 4
	vec = np.zeros(action_n)
	vec[action] = 1
	return torch.tensor(vec).float()
	