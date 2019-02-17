#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
from collections import defaultdict, deque
import random
import argparse

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()

		self.initEpsilon = epsilon
		self.setEpsilon(epsilon)
		self.discountFactor = discountFactor

		# dictionary with automatically assigned default value for a new key
		self.Q = defaultdict(lambda: initVals)

		self.experienceQueue = deque([])

		self.possibleActions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 
			'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']

	def firstVisit(self, state, action, queue):
		# checks whether state action pair is the only one in the experience queue
		for i in range(len(queue)):
			s, a, r = queue[i]
			if state == s and action == a:
				return False
		return True 

	def learn(self):

		G = 0
		while len(self.experienceQueue) > 0:
			state, action, reward = self.experienceQueue.pop()
			G = self.discountFactor * G + reward

			if not self.firstVisit(state, action, self.experienceQueue):
				continue

			Qkey = self.key(state, action)
			# sugalvot, kaip daryt rolling average
			self.Q[Qkey] = 




	def toStateRepresentation(self, state):
		self.currState = state

	def setExperience(self, state, action, reward, status, nextState):
		self.experienceQueue.append((state, action, reward))

	def setState(self, state):
		self.currState = state

	def reset(self):
		raise NotImplementedError

	def greedyAction(self, state):
		# finds an action with the biggest Q value for the state
		# returns (action, QValue) pair

		maxQ = None
		maxAction = None
		for action in self.possibleActions:
			QValue = self.Q[self.key(state, action)]
			# should I worry about equal Q values?
			if maxQ is None or QValue > maxQ:
				maxQ = QValue
				maxAction = action

		return maxAction

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
		decay_constant = 0.0035
		e = 2.718

		factor = e ** (- decay_constant * episode)

		learningRate = self.initLearningRate * factor
		epsilon = self.initEpsilon * factor

		return learningRate, epsilon

	def key(self, state, action):
		return "state: {0}, action: {1}".format(state, action)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, 
		numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	# Run training Monte Carlo Method
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