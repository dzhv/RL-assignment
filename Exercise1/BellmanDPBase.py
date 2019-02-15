from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self, discountRate=0.9):
		self.MDP = MDP()
		self.discountRate = discountRate
		self.initVs()

	def initVs(self):		
		self.Vs = dict()
		self.policy = dict()
		for state in self.MDP.S:
			self.Vs[state] = 0
			self.policy[state] = self.MDP.A


	def action_return(self, state, action):		
		# for each next state:
		# get the state probability given current state and action
		# get the reward for the s, r, s' combination
		# sum the s, r, s' rewards by weighting them by their probability

		state_prob = self.MDP.probNextStates(state, action)

		expected_reward = 0
		for next_state in state_prob:
			prob = state_prob[next_state]
			reward = self.MDP.getRewards(state, action, next_state)
			expected_reward += prob * (reward + self.discountRate * self.Vs[next_state])

		return expected_reward


	def max_action_return(self, state):
		# finds actions with the heighest expected reward
		# and returns the action and its expected reward

		max_return = None
		best_actions = []
		for action in self.MDP.A:  # is every action possible any time?
			# get expected return for the action
			a_return = self.action_return(state, action)
			if max_return is None or max_return < a_return:
				max_return = a_return
				best_actions = [action]
			elif max_return == a_return:
				best_actions.append(action)

		return best_actions, max_return


	def BellmanUpdate(self):		
		for state in self.MDP.S:
			self.policy[state], self.Vs[state] = self.max_action_return(state)

		return self.Vs, self.policy
		
		
if __name__ == '__main__':
	solution = BellmanDPSolver(1)
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values, "\n")	
	print("Policy : ", policy)

