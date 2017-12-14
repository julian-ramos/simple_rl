#!/usr/bin/env python

# Python imports.
import sys
import logging

# Other imports.
import srl_example_setup
from simple_rl.agents import LinearQLearnerAgent, RandomAgent,LinearSarsaAgent,GradientBoostingAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    # Gym MDP
    gym_mdp = GymMDP(env_name='CartPole-v0', render=False,interaction_features=False)
#     gym_mdp = GymMDP(env_name='MountainCar-v0', render=False,interaction_features=True)
    num_feats = gym_mdp.get_num_state_feats()

    # Setup agents and run.
    lin_agent = LinearQLearnerAgent(gym_mdp.actions, num_features=num_feats, alpha=0.3, epsilon=0.3, anneal=True,rbf=False)
    rand_agent = RandomAgent(gym_mdp.actions)
    ls_agent = LinearSarsaAgent(gym_mdp.actions,num_features=num_feats,anneal=True,rbf=True)
#     gd_agent = GradientBoostingAgent(gym_mdp.actions)
    run_agents_on_mdp([lin_agent, rand_agent,ls_agent], gym_mdp, instances=10, episodes=1000, steps=10000, open_plot=open_plot,cumulative_plot=False)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    # logging.basicConfig(level=logging.WARNING)
    main(open_plot=not(sys.argv[-1] == "no_plot"))
