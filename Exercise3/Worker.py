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

import os

policy = Policy()

def increment_counter(counter):
	with counter.get_lock():
		counter.value += 1
		return counter.value

def train(idx, networks, optimizer, counter, environment, policy, config):

	counterValue = 0
	for i in range(config["numEpisodes"]):
		print("\nEpisode {0}/{1}\n".format(i, config["numEpisodes"]))
		print("Counter: {0}".format(counterValue))

		experienceQueue = deque([])
		# gal dar kažką reik resetint?
		# kokius gradient, optimizer ka nors
		environment.reset()

		while True:
			counterValue = increment_counter(counter)

			state = environment.curState
			action, q_value = policy.egreedyAction(state, networks["target"], computePrediction)
			nextState, reward, done, status, info = environment.step(environment.possibleActions[action])
			experienceQueue.append((state, action, reward, q_value, nextState))

			if done or len(experienceQueue) >= config["learning_network_update_interval"]:
				loss = 0 
				for i in range(len(experienceQueue)):
					state, action, reward, q_value, nextState = experienceQueue.popleft()

					target = computeTargets(reward, nextState, config["discountFactor"], done, networks["target"])
					loss += (target - computePrediction(state, action, networks["learning"]))**2

				optimizer.zero_grad()	
				loss.backward()
				
				optimizer.step()			

			if counterValue % config["target_network_update_interval"] == 0:
				networks["target"].load_state_dict(networks["learning"].state_dict())

			if counterValue % config["parameter_save_frequency"] == 0:
				print("Saving target network, counter: {0}".format(counterValue))
				saveModelNetwork(networks["target"], 
					config["parameterStoragePath"] + str(counterValue // config["parameter_save_frequency"]) + ".out")

			if done:
				break


	# save the final model weights
	if idx == 0:	# check the id so that only 1 worker would perform the saving
		print("Saving final target network, counter: {0}".format(counterValue))
		saveModelNetwork(networks["target"], 
			config["parameterStoragePath"] + str(counterValue // config["parameter_save_frequency"]) + ".out")

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	nextObservation = nextObservation.view(-1)
	# get q value of the greedy action
	if done:
		return reward

	_, qmax = policy.greedyAction(nextObservation, targetNetwork, computePrediction)
	return reward + discountFactor * qmax

def computePrediction(state, action, valueNetwork):	
	state = state.view(-1)

	action_tensor = one_hot_encode(action)
	model_input = torch.cat((state, action_tensor))
	
	return valueNetwork(model_input)
	
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)

def one_hot_encode(action):
	action_n = 4
	vec = np.zeros(action_n)
	vec[action] = 1
	return torch.tensor(vec).float()
	