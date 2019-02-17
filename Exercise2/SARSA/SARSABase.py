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

		self.initLearningRate = learningRate
		self.setLearningRate(learningRate)
		self.initEpsilon = epsilon
		self.setEpsilon(epsilon)
		self.discountFactor = discountFactor

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

	# gives the next action in an epsilon-greedy fashion
	def policy(self, state):
		explore = random.random() < self.currEpsilon
		if explore:
			return self.randomAction()

		return self.greedyAction(state)


	def learn(self):
		if len(self.experienceQueue < 2):
			print("\nUh oh, this was not supposed to happen!")
			return 0

		# learning is done for the previous state
		previousState, previousAction, previousReward = self.experienceQueue.popleft()
		currState, currAction, currReward = self.experienceQueue.popleft()
		self.experienceQueue.append((currState, currAction, currReward))


		Qkey = self.key(currState, currAction)
		initialQVal = self.Q[Qkey]

		nextAction = self.policy(self.nextState)
		nextQValue = self.Q[self.key(self.nextState, nextAction)]

		reward = 0 if self.reward is None else self.reward
		target = reward + self.discountFactor * nextQValue
		self.Q[Qkey] = initialQVal + self.currLearningRate * (target - initialQVal)

		return self.Q[Qkey] - initialQVal


	def act(self):
		return self.policy(self.currState)

	def setState(self, state):
		self.currState = state

	def setExperience(self, state, action, reward, status, nextState):
		self.currState = state
		self.action = action
		self.reward = reward
		self.nextState = nextState

	def toStateRepresentation(self, state):
		return str(state)

	def reset(self):
		return

	def computeHyperparameters(self, numTakenActions, episode):
		decay_constant = 0.003
		e = 2.718

		factor = e ** (- decay_constant * episode)

		learningRate = self.initLearningRate * factor
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
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent  (learningRate, discountFactor, epsilon)
	agent = SARSAAgent(0.2, 0.99, 0.25)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):
		print("EPISODE: {0}/{1}".format(episode, numEpisodes))

		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)

			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
				agent.toStateRepresentation(nextObservation))
			
			if not epsStart:
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	print(agent.Q)

	
