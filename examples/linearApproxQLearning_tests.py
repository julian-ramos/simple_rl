#!/usr/bin/env python

# Python imports.
import sys
import logging

# Other imports.
import srl_example_setup
from simple_rl.agents import LinearQLearnerAgent, RandomAgent,QLearnerAgent
from simple_rl.tasks import GymMDP
from simple_rl.tasks import GridWorldMDP

from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
#     gym_mdp = GridWorldMDP(width=10, height=10, init_loc=(1,1), goal_locs=[(10,10)])
#     num_feats = gym_mdp.get_num_state_feats()
#     lin_agent = QLearnerAgent(gym_mdp.actions, alpha=0.4, epsilon=0.4)
#     rand_agent = RandomAgent(gym_mdp.actions)
#     run_agents_on_mdp([lin_agent, rand_agent], gym_mdp, instances=50, episodes=200, steps=100, open_plot=open_plot)
    
    
#     gym_mdp = GridWorldMDP(width=10, height=10, init_loc=(1,1), goal_locs=[(10,10)])
#     num_feats = gym_mdp.get_num_state_feats()
#     lin_agent = LinearQLearnerAgent(gym_mdp.actions, num_features=num_feats, alpha=0.4, epsilon=0.4, anneal=False,rbf=True)
#     rand_agent = RandomAgent(gym_mdp.actions)
#     run_agents_on_mdp([lin_agent, rand_agent], gym_mdp, instances=50, episodes=200, steps=100, open_plot=open_plot,verbose=True)
    
    gym_mdp = GymMDP(env_name='CartPole-v0', render=True)
    num_feats = gym_mdp.get_num_state_feats()
    lin_agent = LinearQLearnerAgent(gym_mdp.actions, num_features=num_feats, alpha=0.4, epsilon=0.4, anneal=False,rbf=True)
    rand_agent = RandomAgent(gym_mdp.actions)
    run_agents_on_mdp([lin_agent, rand_agent], gym_mdp, instances=5, episodes=1000, steps=100, open_plot=open_plot)
    
    
    
 

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    # logging.basicConfig(level=logging.WARNING)
    main(open_plot=not(sys.argv[-1] == "no_plot"))
