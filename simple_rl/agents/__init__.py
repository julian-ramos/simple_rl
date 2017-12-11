'''
Implementations of standard RL agents:

	AgentClass: Contains the basic skeleton of an RL Agent.
	QLearnerAgentClass: QLearner.
	LinearQLearnerAgentClass: Q Learner with a Linear Approximator.
	RandomAgentClass: Random actor.
	RMaxAgentClass: RMax.
	LinUCBAgentClass: Conextual Bandit Algorithm.
'''

# Grab classes.
from AgentClass import Agent
from FixedPolicyAgentClass import FixedPolicyAgent
from QLearnerAgentClass import QLearnerAgent
from RandomAgentClass import RandomAgent
from RMaxAgentClass import RMaxAgent
from DataQLearnerAgentClass import DataQLearnerAgent
from func_approx.LinearQLearnerAgentClass import LinearQLearnerAgent
from func_approx.LinearSarsaAgentClass import LinearSarsaAgent
from func_approx.DataLinearQLearnerAgentClass import DataLinearQLearnerAgent

from bandits.LinUCBAgentClass import LinUCBAgent