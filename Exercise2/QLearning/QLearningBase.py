#!/usr/bin/env python3
# encoding utf-8

import sys
from os import path

this_folder = path.dirname(path.abspath(__file__))
sys.path.append( path.dirname(this_folder) )

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
from collections import defaultdict
import random
import argparse


class QLearningAgent(Agent):
	def __init__(self, learningRate=0.1, discountFactor=0.95, epsilon=1, initVals=0.0):
		super(QLearningAgent, self).__init__()
		
		self.initLearningRate = learningRate
		self.setLearningRate(self.initLearningRate)
		self.initEpsilon = epsilon
		self.setEpsilon(self.initEpsilon)
		self.discountFactor = discountFactor
		self.decay_constant = 0.0006

		# dictionary with automatically assigned default value for a new key
		self.Q = defaultdict(lambda: initVals)

		self.possibleActions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 
			'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']

	def key(self, state, action):
		return "state: {0}, action: {1}".format(state, action)

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

	def policy(self, state):
		# gives the next action in an epsilon-greedy fashion
		explore = random.random() < self.currEpsilon
		if explore:
			return self.randomAction()

		return self.greedyAction(state)

	def learn(self):
		Qkey = self.key(self.currState, self.action)
		initialQVal = self.Q[Qkey]
				
		target = self.reward + self.discountFactor * self.maxQ(self.nextState)
		self.Q[Qkey] = initialQVal + self.currLearningRate * (target - initialQVal)

		return self.Q[Qkey] - initialQVal

	def act(self):
		return self.policy(self.currState)

	def toStateRepresentation(self, state):
		return str(state)

	def setState(self, state):
		self.currState = state

	def setExperience(self, state, action, reward, status, nextState):
		self.currState = state
		self.action = action
		self.reward = reward
		self.nextState = nextState

	def setLearningRate(self, learningRate):
		self.currLearningRate = learningRate

	def setEpsilon(self, epsilon):
		self.currEpsilon = epsilon

	def reset(self):
		self.experienceQueue.clear()
		
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
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)
	parser.add_argument('--learningRate', type=float, default=0.1)
	parser.add_argument('--discountFactor', type=float, default=0.1)
	parser.add_argument('--decayConstant', type=float, default=0.0006)
	parser.add_argument('--experiment', type=str, default="exp1")

	args=parser.parse_args()
	

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, 
		agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate=args.learningRate, discountFactor=args.discountFactor, 
		epsilon=1, decay_constant=args.decayConstant)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	goals = 0
	for episode in range(numEpisodes):

		status = 0
		observation = hfoEnv.reset()
				
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)

			if reward > 0:
				goals += 1
				
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, 
				agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation	