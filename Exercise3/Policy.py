import random


class Policy():
	def __init__(self, epsilon=1):
		self.epsilon = epsilon
		self.possibleActions = ['MOVE','SHOOT','DRIBBLE','GO_TO_BALL']

	def randomAction(self):
		actionIndex = random.randint(0, len(self.possibleActions) - 1)
		return actionIndex

	def greedyAction(self, state, valueNetwork, computePrediction):
		maxQ = None
		maxAction = None
		for action in range(len(self.possibleActions)):
			QValue = computePrediction(state, action, valueNetwork)
			# should I worry about equal Q values?
			if maxQ is None or QValue > maxQ:
				maxQ = QValue
				maxAction = action

		return maxAction, maxQ

	def egreedyAction(self, state, valueNetwork, computePrediction):
		# gives the next action in an epsilon-greedy fashion
		explore = random.random() < self.epsilon
		if explore:
			action = self.randomAction()
			prediction = computePrediction(state, action, valueNetwork)
			return action, prediction

		return self.greedyAction(state, valueNetwork, computePrediction)

