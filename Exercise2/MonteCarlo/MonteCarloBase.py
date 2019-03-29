#!/usr/bin/env python3
# encoding utf-8

import sys
from os import path

this_folder = path.dirname(path.abspath(__file__))
sys.path.append( path.dirname(this_folder) )

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
from collections import defaultdict, deque
import random
import argparse

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor=0.99, epsilon=1, initVals=0.0):
		super(MonteCarloAgent, self).__init__()

		self.initEpsilon = epsilon
		self.setEpsilon(self.initEpsilon)
		self.discountFactor = discountFactor

		# dictionary with automatically assigned default value for a new key
		self.Q = defaultdict(lambda: initVals)
		# dictionary wiht counts of visits for each state-action pair
		# used for computing iterative average of returns 
		self.stateActionVisits = defaultdict(lambda: initVals)

		self.experienceQueue = deque([])

		self.possibleActions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 
			'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']

	def firstVisit(self, state, action, queue):
		# checks whether state action pair exists in the experience queue
		# if it does not exists, this is the first visit of state action
		for i in range(len(queue)):
			s, a, r = queue[i]
			if state == s and action == a:
				return False
		return True 

	def learn(self):
		G = 0
		resultQueue = deque([])
		while len(self.experienceQueue) > 0:
			state, action, reward = self.experienceQueue.pop()
			G = self.discountFactor * G + reward

			if not self.firstVisit(state, action, self.experienceQueue):
				continue

			key = self.key(state, action)
			
			n = self.stateActionVisits[key]
			self.stateActionVisits[key] += 1
						
			self.Q[key] = self.Q[key] * (n / (n+1)) + G / (n+1)
			
			resultQueue.appendleft(self.Q[key])
		
		return self.Q, list(resultQueue)

	def toStateRepresentation(self, state):
		return str(state)

	def setExperience(self, state, action, reward, status, nextState):
		self.experienceQueue.append((state, action, reward))

	def setState(self, state):
		self.currState = state

	def reset(self):
		self.experienceQueue.clear()

	def greedyAction(self, state):
		# finds an action with the biggest Q value for the state		

		maxQ = None
		maxAction = None
		for action in self.possibleActions:
			QValue = self.Q[self.key(state, action)]
			if maxQ is None or QValue > maxQ:
				maxQ = QValue
				maxAction = action

		return maxAction

	def randomAction(self):
		actionIndex = random.randint(0, len(self.possibleActions) - 1)
		return self.possibleActions[actionIndex]

	def policy(self, state):
		# gives the next action in an epsilon-greedy fashion
		explore = random.random() < self.currEpsilon
		if explore:
			return self.randomAction()

		return self.greedyAction(state)

	def act(self):
		return self.policy(self.currState)

	def setEpsilon(self, epsilon):
		self.currEpsilon = epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# exponential decay
		decay_constant = 0.0006
		e = 2.718

		factor = e ** (- decay_constant * episode)
		
		epsilon = self.initEpsilon * factor

		return epsilon

	def key(self, state, action):
		return "state: {0}, action: {1}".format(state, action)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)
	parser.add_argument('--experiment', type=str, default="exp_test")

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, 
		numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()
	
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0	
	
	goals = 0
	for episode in range(numEpisodes):
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)			

			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, 
				status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()