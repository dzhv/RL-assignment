from Worker import *

def run(num_episodes, network, environment, policy, logger, counter):
	goals = 0
	last_counter_value = -1
	counter_checks = 0
	for i in range(num_episodes):
		# logger.log("Episode {0}/{1}".format(i, num_episodes))
		environment.reset()

		while True:
			state = environment.curState
			action, q_value = policy.greedyAction(computePredictions(state, network))

			# logger.log(str(networks["learning"].model[2].weight))

			nextState, reward, done, status, info = environment.step(environment.possibleActions[action])

			if reward == 1:
				goals += 1

			if status == 5:
				logger.log("Something is wrong, getting Status 5")

			if done:
				break

		if i != 1 and i % 100 == 0:
			logger.log("Goals over 100 last episodes: {0}".format(goals))
			goals = 0
		
		if not counter is None:
			if last_counter_value == counter.value:
				counter_checks += 1
			else:
				counter_checks = 1

			if counter_checks > 30:
				logger.log("seems like everyone else has stopped")
				logger.log("stopping")
				break

			last_counter_value = counter.value
