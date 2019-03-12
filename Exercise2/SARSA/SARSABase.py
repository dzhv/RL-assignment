#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
from collections import defaultdict, deque
import random

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()

		self.initLearningRate = 0.1
		self.setLearningRate(learningRate)
		self.initEpsilon = 1
		self.setEpsilon(epsilon)
		self.discountFactor = 0.99

		# dictionary with automatically assigned default value for a new key
		self.Q = defaultdict(lambda: initVals)

		# this is a queue holding previous and current (s, a, r) values
		# for the learn() function
		self.experienceQueue = deque([])

		self.possibleActions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 
			'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']

	def key(self, state, action):
		return "state: {0}, action: {1}".format(state, action)

	def randomAction(self):
		actionIndex = random.randint(0, len(self.possibleActions) - 1)
		return self.possibleActions[actionIndex]

	def greedyAction(self, state):
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


	def learn(self):
		if len(self.experienceQueue) < 2:
			return 0

		# learning is done for the previous state
		prevState, prevAction, prevReward = self.experienceQueue.popleft()
		currState, currAction, currReward = self.experienceQueue.popleft()
		self.experienceQueue.append((currState, currAction, currReward))

		prevKey = self.key(prevState, prevAction)
		currKey = self.key(currState, currAction)
		initialQVal = self.Q[prevKey]
		nextQValue = self.Q[currKey]
		
		target = prevReward + self.discountFactor * nextQValue
		self.Q[prevKey] = initialQVal + self.currLearningRate * (target - initialQVal)

		# print("Prev state: {0}".format(prevState))
		# print("Prev reward: {0}".format(prevReward))
		# print("Prev action: {0}".format(prevAction))
		# print("Current state: {0}".format(currState))
		# print("Current reward: {0}".format(currReward))
		# print("Current action: {0}".format(currAction))
		# print("Update: {0} -> {1}".format(initialQVal, self.Q[prevKey]))

		return self.Q[prevKey] - initialQVal

	def act(self):
		return self.policy(self.currState)

	def setState(self, state):
		self.currState = state

	def setExperience(self, state, action, reward, status, nextState):
		self.experienceQueue.append((state, action, reward))

	def toStateRepresentation(self, state):
		return str(state)

	def reset(self):
		self.experienceQueue.clear()

	def computeHyperparameters(self, numTakenActions, episode):

		decay_constant = 0.0006
		e = 2.718

		factor = e ** (- decay_constant * episode)

		learningRate = self.initLearningRate# * factor
		epsilon = self.initEpsilon * factor

		return learningRate, epsilon

	def setLearningRate(self, learningRate):
		self.currLearningRate = learningRate

	def setEpsilon(self, epsilon):
		self.currEpsilon = epsilon

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, 
		numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent  (learningRate, discountFactor, epsilon)
	agent = SARSAAgent(learningRate=0.1, discountFactor=0.99, epsilon=1)

	# Run training using SARSA
	numTakenActions = 0 
	goals = 0
	for episode in range(numEpisodes):
		print("EPISODE: {0}/{1}".format(episode, numEpisodes))

		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True


		if episode % 100 == 0:
			print("Goals last 100 episodes: {0}".format(goals))
			goals = 0
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
			# print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
				agent.toStateRepresentation(nextObservation))
			
			if not epsStart:
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	# print(agent.Q)

	
