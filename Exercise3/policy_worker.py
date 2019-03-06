from Worker import *

def run(num_episodes, value_network, environment, policy, logger):
	for i in range(num_episodes):
		logger.log("\nEpisode {0}/{1}\n".format(i, num_episodes))
		environment.reset()

		while True:
			state = environment.curState
			action, q_value = policy.greedyAction(state, value_network, computePrediction)
			nextState, reward, done, status, info = environment.step(environment.possibleActions[action])

			if done:
				break