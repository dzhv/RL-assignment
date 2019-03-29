#!/usr/bin/env python3
# encoding utf-8

import sys
from os import path
sys.path.append( path.dirname( path.dirname( __file__ ) ) )

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
from collections import defaultdict
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate=0.15, discountFactor=0.99, winDelta=0.001, loseDelta=0.01, initVals=0.0):
		super(WolfPHCAgent, self).__init__()

		self.initLearningRate = learningRate
		self.setLearningRate(self.initLearningRate)
		self.initWinDelta = winDelta
		self.setWinDelta(self.initWinDelta)
		self.initLoseDelta = loseDelta
		self.setLoseDelta(self.initLoseDelta)
		self.discountFactor = discountFactor
		self.delta_update = 0.00000035

		# Best:    lr= 0.15, dc = 0.00025  df = 0.95 winDelta=0.01 loseDelta=0.1   |   608

		self.possibleActions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK', 'NO_OP']

		# dictionary with automatically assigned default value for a new key
		self.Q = defaultdict(lambda: initVals)
		# policy values
		self.Pi = defaultdict(lambda: 1 / len(self.possibleActions))
		self.avg_pi = defaultdict(lambda: 1 / len(self.possibleActions))
		# state counts
		self.C = defaultdict(lambda: initVals)
		
	def setExperience(self, state, action, reward, status, nextState):
		self.currState = state
		self.action = action
		self.reward = reward
		self.nextState = nextState

	def action_probabilities(self, policy, state):
		# returns probabilities for all the actions under a policy and given a state
		return [policy[self.key(state, action)] for action in self.possibleActions]

	def act(self):
		# sample an action from the policy
		probabilities = self.action_probabilities(self.Pi, self.currState)
		action = np.random.choice(self.possibleActions, size=1, p=probabilities)
		return action[0]

	def key(self, state, action):
		return "state: {0}, action: {1}".format(state, action)

	def learn(self):
		Qkey = self.key(self.currState, self.action)
		initialQVal = self.Q[Qkey]
				
		target = self.reward + self.discountFactor * self.maxQ(self.nextState)
		self.Q[Qkey] = initialQVal + self.currLearningRate * (target - initialQVal)

		return self.Q[Qkey] - initialQVal

	def greedyAction(self, state):
		# gets greedy action
		return self.maxQAction(state)[0]

	def maxQ(self, state):
		# gets the Q value of a greedy action
		return self.maxQAction(state)[1]

	def maxQAction(self, state):
		# finds an action with the biggest Q value for the state
		# returns (action, QValue) pair
		maxQ = None
		maxAction = None
		for action in self.possibleActions:
			QValue = self.Q[self.key(state, action)]
			if maxQ is None or QValue > maxQ:
				maxQ = QValue
				maxAction = action

		return maxAction, maxQ

	def calculateAveragePolicyUpdate(self):
		self.C[self.currState] += 1
		for action in self.possibleActions:
			act_key = self.key(self.currState, action)
			self.avg_pi[act_key] += 1 / self.C[self.currState] * \
				(self.Pi[act_key] - self.avg_pi[act_key])

		return self.action_probabilities(self.avg_pi, self.currState)

	def calculatePolicyUpdate(self):
		_, maxQ = self.maxQAction(self.currState)
		optimal_actions = [a for a in self.possibleActions if self.Q[self.key(self.currState, a)] == maxQ]
		suboptimal_actions = [a for a in self.possibleActions if self.Q[self.key(self.currState, a)] < maxQ]

		if len(optimal_actions) + len(suboptimal_actions) != len(self.possibleActions):
			raise Exception("Some actions were lost somehow?")

		delta = self.winDelta if self.agent_is_winning() else self.loseDelta

		p_moved = 0
		for action in suboptimal_actions:
			policy_prob = self.Pi[self.key(self.currState, action)]
			update = min(delta / len(suboptimal_actions), policy_prob)
			p_moved += update
			self.Pi[self.key(self.currState, action)] -= update

		for action in optimal_actions:
			self.Pi[self.key(self.currState, action)] += \
				p_moved / (len(self.possibleActions) - len(suboptimal_actions))

		return self.action_probabilities(self.Pi, self.currState)

	def agent_is_winning(self):
		sum_pi = 0
		sum_avg_pi = 0 
		for action in self.possibleActions:
			act_key = self.key(self.currState, action)
			sum_pi += self.Pi[act_key] * self.Q[act_key]
			sum_avg_pi += self.avg_pi[act_key] * self.Q[act_key]

		return sum_pi >= sum_avg_pi
	
	def toStateRepresentation(self, state):
		return str(state)

	def setState(self, state):
		self.currState = state

	def setLearningRate(self, lr):
		self.currLearningRate = lr
		
	def setWinDelta(self, winDelta):
		self.winDelta = winDelta
		
	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		loseDelta = min(0.9, self.initLoseDelta + episodeNumber * self.delta_update)
		winDelta = min(1, self.initWinDelta + episodeNumber * self.delta_update)

		return loseDelta, winDelta, self.initLearningRate

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent()
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	total_reward = 0
	reward_last_1000 = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()

		if episode % 1000 == 0:
			print("Reward last 1000 episodes: {0}".format(reward_last_1000))
			reward_last_1000 = 0
			print("Deltas: winning {0}, lose {1}".format(agents[0].winDelta, agents[0].loseDelta))
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1			

			total_reward += reward[0]
			reward_last_1000 += reward[0] 

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], 
					reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			
			observation = nextObservation

	print("Average reward over 1000 episodes: {0}".format(total_reward * 1000 / numEpisodes))
