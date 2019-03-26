#!/usr/bin/env python3
#encoding utf-8

from hfo import *
from copy import copy, deepcopy
import math
import random
import os
import time
import torch

class HFOEnv(object):

	def __init__(self, config_dir = '../../../bin/teams/base/config/formations-dt', 
		port = 6000, server_addr = 'localhost', team_name = 'base_left', play_goalie = False,
		numOpponents = 0, numTeammates = 0, seed = 123):

		self.config_dir = config_dir
		self.port = port
		self.server_addr = server_addr
		self.team_name = team_name
		self.play_goalie = play_goalie

		self.curState = None
		self.possibleActions = ['MOVE','SHOOT','DRIBBLE','GO_TO_BALL']
		self.numOpponents = numOpponents
		self.numTeammates = numTeammates
		self.seed = seed
		self.startEnv()
		self.hfo = HFOEnvironment()
		

	# Method to initialize the server for HFO environment
	def startEnv(self):
		if self.numTeammates == 0:
			os.system("./../../../bin/HFO --seed {} --defense-npcs=0 --headless --defense-agents={} --offense-agents=1 --trials 30000 --untouched-time 500 --frames-per-trial 500 --port {} --fullstate &".format(str(self.seed),
				str(self.numOpponents), str(self.port)))
		else :
			os.system("./../../../bin/HFO --seed {} --defense-agents={} --defense-npcs=0 --offense-npcs={} --offense-agents=1 --trials 30000 --untouched-time 500 --frames-per-trial 500 --port {} --fullstate &".format(
				str(self.seed), str(self.numOpponents), str(self.numTeammates), str(self.port)))
		time.sleep(5)

	# Reset the episode and returns a new initial state for the next episode
	# You might also reset important values for reward calculations
	# in this function
	def reset(self):
		processedStatus = self.preprocessState(self.hfo.getState())
		self.curState = processedStatus
 
		return self.curState

	# Connect the custom weaker goalkeeper to the server and 
	# establish agent's connection with HFO server
	def connectToServer(self):
		os.system("./Goalkeeper.py --numEpisodes=30000 --port={} &".format(str(self.port)))
		time.sleep(2)
		self.hfo.connectToServer(LOW_LEVEL_FEATURE_SET,self.config_dir,self.port,self.server_addr,self.team_name,self.play_goalie)

	# This method computes the resulting status and states after an agent decides to take an action
	def act(self, actionString):

		if actionString =='MOVE':
			self.hfo.act(MOVE)
		elif actionString =='SHOOT':
			self.hfo.act(SHOOT)
		elif actionString =='DRIBBLE':
			self.hfo.act(DRIBBLE)
		elif actionString =='GO_TO_BALL':
			self.hfo.act(GO_TO_BALL)
		else:
			raise Exception('INVALID ACTION!')

		status = self.hfo.step()
		currentState = self.hfo.getState()
		processedStatus = self.preprocessState(currentState)
		self.curState = processedStatus

		return status, self.curState

	# Method that serves as an interface between a script controlling the agent
	# and the environment. Method returns the nextState, reward, flag indicating
	# end of episode, and current status of the episode
	def step(self, action_params):
		status, nextState = self.act(action_params)
		done = (status!=IN_GAME)
		reward, info = self.get_reward(status, nextState)
		return nextState, reward, done, status, info

	# This method enables agents to quit the game and the connection with the server
	# will be lost as a result
	def quitGame(self):
		self.hfo.act(QUIT)

	# Preprocess the state representation in this function
	def preprocessState(self, state):
		return torch.tensor([state]).float()

	# Define the rewards you use in this function
	# You might also give extra information on the name of each rewards
	# for monitoring purposes.
	def get_reward(self, status, nextState):
		# print("Current State, shape: {0}".format(self.curState.shape))
		# print(self.curState)
		# print("next State")
		# print(nextState)



		reward = 0.0
		info = {}
		# print("GOAL value: {0}".format(GOAL))    ==  1		
		# print("CAPTURED_BY_DEFENSE value: {0}".format(CAPTURED_BY_DEFENSE)) == 2
		# print("OUT_OF_BOUNDS value: {0}".format(OUT_OF_BOUNDS))   ==  3

		if status == GOAL:
			reward = 1
			info = { "reason": "GOAL" }
		elif status == CAPTURED_BY_DEFENSE:
			reward = -0.1
			info = { "reason": "CAPTURED_BY_DEFENSE" }
		elif status == OUT_OF_BOUNDS:
			reward = -0.2
			info = { "reason": "Out of bounds" }
		else:
			info = { "reason": "Nothing happened"}
		
		# 		print("Status: {0}, Reward: {1} granted. Reason: {2}".format(status, reward, info))
		return reward, info







