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
from collections import defaultdict
		
class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate=0.1, discountFactor=0.9, epsilon=1, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()

		self.initLearningRate = learningRate
		self.setLearningRate(self.initLearningRate)
		self.initEpsilon = epsilon
		self.setEpsilon(self.initEpsilon)
		self.discountFactor = discountFactor
		self.decay_constant = 0.00025

		self.possibleActions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK', 'NO_OP']

		# dictionary with automatically assigned default value for a new key
		self.Q = defaultdict(lambda: initVals)

	def setExperience(self, state, action, reward, status, nextState):
		self.currState = state
		self.action = action
		self.reward = reward
		self.nextState = nextState

	def key(self, state, action):
		return "state: {0}, action: {1}".format(state, action)
	
	def learn(self):
		Qkey = self.key(self.currState, self.action)
		initialQVal = self.Q[Qkey]
				
		target = self.reward + self.discountFactor * self.maxQ(self.nextState)
		self.Q[Qkey] = initialQVal + self.currLearningRate * (target - initialQVal)

		return self.Q[Qkey] - initialQVal

	def act(self):
		return self.policy(self.currState)

	def policy(self, state):
		# gives the next action in an epsilon-greedy fashion
		explore = random.random() < self.currEpsilon
		if explore:
			return self.randomAction()

		return self.greedyAction(state)

	def randomAction(self):
		actionIndex = random.randint(0, len(self.possibleActions) - 1)
		return self.possibleActions[actionIndex]

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

	def greedyAction(self, state):
		# gets greedy action
		return self.maxQAction(state)[0]

	def maxQ(self, state):
		# gets the Q value of a greedy action
		return self.maxQAction(state)[1]

	def toStateRepresentation(self, state):
		return str(state)

	def setState(self, state):
		self.currState = state

	def setLearningRate(self, learningRate):
		self.currLearningRate = learningRate

	def setEpsilon(self, epsilon):
		self.currEpsilon = epsilon
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# exponential decay
		decay_constant = self.decay_constant
		e = 2.718

		factor = e ** (- decay_constant * episode)

		learningRate = self.initLearningRate# * factor
		epsilon = self.initEpsilon * factor

		return learningRate, epsilon

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent()
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	total_reward = 0
	reward_last_1000 = 0
	for episode in range(numEpisodes):	
		# print("Episode: {0}".format(episode))

		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()

		if episode % 1000 == 0:
			print("Reward last 1000 episodes: {0}".format(reward_last_1000))
			reward_last_1000 = 0

		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)			
			
			if reward[0] != reward[1]:
				raise Exception("ehm what? reward[0] != reward[1]")

			total_reward += reward[0]
			reward_last_1000 += reward[0] 

			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), 
					actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation

	print("Average reward over 1000 episodes: {0}".format(total_reward * 1000 / numEpisodes))