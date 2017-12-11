#!/usr/bin/env python

# Python imports.
import sys
import logging

# Other imports.
import srl_example_setup
from simple_rl.agents import DataLinearQLearnerAgent, RandomAgent
from simple_rl.tasks import DataMDP

from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    
    
    state_features=['physical','sleep_quality','month','sleepDurationDeviation','weekday','deadlines','social','nightQuietness']
    action_features=['act_sleep_hours']
    reward_feature=['sleep_quality']
    continuity_feature=['dayOfYear']
    

    data_mdp = DataMDP(user='aiproject',hostname='julian.hcii.cs.cmu.edu',
                       db_name='studentLife_tz',passwd='Hcirulz@85',
                       render=False,
                       tablename='studentLife_tz.fixedStandard where user!=17 and user!=18 and user!=23 and user!=33 and user!=46 and user!=52',
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
    run_agents_on_mdp([data_lin_agent,rand_agent], data_mdp, instances=1, episodes=103, steps=1000, open_plot=open_plot,verbose=True)
    

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    # logging.basicConfig(level=logging.WARNING)
    main(open_plot=not(sys.argv[-1] == "no_plot"))
