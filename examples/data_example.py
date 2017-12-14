#!/usr/bin/env python

# Python imports.
import sys
import logging

# Other imports.
import srl_example_setup
from simple_rl.agents import DataLinearQLearnerAgent, RandomAgent
from simple_rl.tasks import DataMDP

from simple_rl.run_experiments import run_agents_on_mdp
import yaml

# config =yaml.load(open('./configurations/config.yaml'))
config =yaml.load(open('./configurations/main.yaml'))

def main(open_plot=True):
    
    state_features=config['state_features']
    action_features=config['action_features']
    reward_feature=config['reward_feature']
    continuity_feature=config['continuity_feature']
    user=config['user']
    hostname=config['hostname']
    db_name=config['db_name']
    passwd=config['passwd']
    tablename=config['tablename']
    
#     state_features=['physical','sleep_quality','month','sleepDurationDeviation','weekday','deadlines','social','nightQuietness']
#     action_features=['act_sleep_hours']
#     reward_feature=['sleep_quality']
#     continuity_feature=['dayOfYear']
    

    data_mdp = DataMDP(user=user,hostname=hostname,
                       db_name=db_name,passwd=passwd,
                       render=False,
                       tablename=tablename,
                       state_features=state_features,
                       action_features=action_features,
                       reward_feature=reward_feature,
                       continuity_feature=continuity_feature)
    
    num_feats = data_mdp.get_num_state_feats()
    '''
    Seems to be working fine up to here, however this was only the initialization code
    Need to check on the actual updates
    '''
    data_lin_agent = DataLinearQLearnerAgent(data_mdp.actions, 
                                             num_features=num_feats,
                                            alpha=0.4, 
                                            epsilon=0.4, 
                                            anneal=False,
                                            rbf=False,
                                            mdp=data_mdp)
#The     
    rand_agent = RandomAgent(data_mdp.actions)
    run_agents_on_mdp([data_lin_agent,rand_agent], 
                      data_mdp, instances=1, episodes=103, steps=1000, 
                      open_plot=open_plot,
                      verbose=True)
    

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    # logging.basicConfig(level=logging.WARNING)
    main(open_plot=not(sys.argv[-1] == "no_plot"))
