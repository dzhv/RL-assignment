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
import itertools
from itertools import combinations_with_replacement as combinations
from collections import defaultdict
		
class JointQLearningAgent(Agent):
	def __init__(self, learningRate=0.15, discountFactor=0.9, epsilon=1, numTeammates=1, initVals=0.0):
		super(JointQLearningAgent, self).__init__()	

		self.initLearningRate = learningRate
		self.setLearningRate(self.initLearningRate)
		self.initEpsilon = epsilon
		self.setEpsilon(self.initEpsilon)
		self.discountFactor = discountFactor
		self.decay_constant = 0.00025

		# Best:    lr= 0.15, dc = 0.00025  df = 0.95   |   694.739

		self.numTeammates = numTeammates
		self.possibleActions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK', 'NO_OP']

		# dictionary with automatically assigned default value for a new key
		self.Q = defaultdict(lambda: initVals)
		# opponent model
		self.C = defaultdict(lambda: initVals)
		self.state_counts = defaultdict(lambda: 0)

	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		self.currState = state
		self.action = action
		self.reward = reward
		self.nextState = nextState
		self.oppoActions = oppoActions

	def key(self, state, actions):		
		return "state: {0}, actions: {1}".format(state, list(actions))
	
	def learn(self):
		self.state_counts[self.currState] += 1
		self.C[self.key(self.currState, self.oppoActions)] += 1

		actions = [self.action] + self.oppoActions

		Qkey = self.key(self.currState, actions)
		initialQVal = self.Q[Qkey]		
				
		target = self.reward + self.discountFactor * self.maxQ(self.nextState)
		self.Q[Qkey] = initialQVal + self.currLearningRate * (target - initialQVal)

		update = self.Q[Qkey] - initialQVal

		return update

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
		# finds an action with the biggest score for the state
		# returns (action, score) pair

		maxScore = None
		maxAction = None
		
		oppo_action_combinations = list(combinations(self.possibleActions, self.numTeammates))

		for action in self.possibleActions:
			score = 0 
			for oppo_actions in oppo_action_combinations:
				actions = [action] + list(oppo_actions)
				QValue = self.Q[self.key(state, actions)]
				# if QValue != 0:
				# 	print("all good")
				score += self.get_actions_estimate(state, oppo_actions) * QValue

			if maxScore is None or score > maxScore:
				maxScore = score
				maxAction = action

		return maxAction, maxScore

	def get_actions_estimate(self, state, oppoActions):
		# computes the estimate probability that opponents will take 
		# actions oppoActions
		# Which is regarded as C(s, a_i-1) / n (s)   in the algorithm
	
		if self.state_counts[state] == 0:
			return 1 / (len(oppoActions) * len(self.possibleActions))

		CValue = self.C[self.key(state, oppoActions)]
		return CValue / self.state_counts[state]

	def greedyAction(self, state):
		# gets greedy action
		return self.maxQAction(state)[0]

	def maxQ(self, state):
		# gets the Q value of a greedy action
		return self.maxQAction(state)[1]

	def toStateRepresentation(self, rawState):
		return str(rawState)

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
		epsilon = self.initEpsilon * factor

		# learningRate = max(0.02, self.initLearningRate - episodeNumber / 1.4e-5)# * factor
		learningRate = self.initLearningRate

		return learningRate, epsilon	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent()
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0

	total_reward = 0
	reward_last_1000 = 0
	for episode in range(numEpisodes):	
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
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			total_reward += reward[0]
			reward_last_1000 += reward[0] 

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), 
					actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation

	print("Average reward over 1000 episodes: {0}".format(total_reward * 1000 / numEpisodes))