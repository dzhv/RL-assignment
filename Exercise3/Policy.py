import random


class Policy():
	def __init__(self, epsilon=1, numUpdates=5000, minEpsilon=0.1, logger=None):
		self.epsilon = epsilon
		self.initEpsilon = epsilon
		self.numUpdates = numUpdates
		self.minEpsilon = minEpsilon
		self.currStep = 0
		self.possibleActions = ['MOVE','SHOOT','DRIBBLE','GO_TO_BALL']
		self.logger = logger

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

	def updateEpsilon(self):
		self.logger.log("Updating epsilon from: {0}".format(self.epsilon))

		self.epsilon = self.initEpsilon - (float(self.currStep) / 
			self.numUpdates * (self.initEpsilon - self.minEpsilon))
		self.epsilon = max(self.epsilon, self.minEpsilon)

		self.logger.log("To: {0}".format(self.epsilon))

class RandomPolicy():
	def __init__(self, epsilon=1):
		self.epsilon = epsilon
		self.possibleActions = [2, 3] # 'DRIBBLE','GO_TO_BALL'

	def randomAction(self):
		actionIndex = random.randint(0, len(self.possibleActions) - 1)
		return actionIndex

	def greedyAction(self, state, valueNetwork, computePrediction):
		return self.randomAction()

	def egreedyAction(self, state, valueNetwork, computePrediction):
		return self.randomAction()