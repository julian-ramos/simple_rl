'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import numpy as np
import random
import sys
import os
import random

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.data.DataStateClass import DataState
from sqlalchemy import create_engine

import pandas as pd


class DataMDP(MDP):
    ''' Class for Simulating an MDP for simple RL using already collected data'''
    def connect(self,database,user,passwd,hostname):
        engine = create_engine('mysql://%s:%s@%s/%s'%(user,passwd,hostname,database))
        return engine.connect()
    
    def get_data_from_db(self,con,tablename):
        '''
        retrieves all the data from the database
        '''
        query="select * from %s order by user, month, day"%(tablename)
        return pd.read_sql(query,con)

    def __init__(self, db_name='',user='',passwd='',hostname='', 
                 render=False,tablename='',state_features=[],
                 action_features=[],reward_feature=[],continuity_feature=[]):
        '''
        This class simulates as if there is an environment and provides a state and reward 
        sequences based on that. There is no exploration for this reason and there is a time
        variable used to keep track of where in the sequence is the MDP. This class should
        be used in tandem with DataLinearQLearningAgent

        Args:
            db_name : Database to be accessed
            user : MySQL user
            passwd : password
            hostname : ip address or other that identifies the server
            tablename : Table in the database where the data resides
            state_features : list of the features (string) that are part of the state
            action_features : list of actions(string) to be used
            reward_feature : list of reward (string) features, currently only supports one reward signal
            continuity_feature : This feature is used to check whether steps are consecutive, so that if a
            given step is 1 and the next is 3, 1 will be treated as a terminal state and will end an episode
            while 3 will be an initial state
        '''
        
        #Database related args
        self.db_name = db_name
        self.state_features=state_features
        self.action_features=action_features
        self.reward_feature=reward_feature
        self.continuity_feature=continuity_feature
        self.con=self.connect(db_name,user,passwd,hostname)
        self.data=self.get_data_from_db(self.con,tablename)
        self.env_name='data'
        # This variable controls where we are at in the sequence
        # it is used also by the AgentClass to retrieve actions
        self.t=0
        
        
        
        self.render = render
        actions=self.data[self.action_features].values
        usersList=self.data['user'].values
        users=np.unique(self.data['user'].values)
        disActs=np.zeros([len(actions),1])
        
        #Discretization step necessary to get Qlearning to work
        for u in users:
            inds=np.argwhere(usersList==u)
            
            fp=np.percentile(actions[inds],33)
            sp=np.percentile(actions[inds],66)
            disActs[inds]=np.digitize(actions[inds],[0,fp,sp,1])
            
        self.allActions=actions
        self.allDiscreteActions=disActs
        
        
        
         
        MDP.__init__(self, np.unique(disActs), 
                     self._transition_func, self._reward_func, 
                     init_state=DataState(self.data[self.state_features].values[0,:]))
        
        
    def step(self):
        
        if self.t+1>self.data.shape[0]-1:
            terminal=True
        else:
            t1=self.data[self.continuity_feature].values[self.t+1,:]
            t0=self.data[self.continuity_feature].values[self.t,:]
            
            if t1-t0>1:
                terminal=True
                #Perhaps 1 after the next it will be a new episode
                if self.t+2<=self.data.shape[0]-2:
                    self.t=self.t+2
                else:
                    raise ValueError("Out of bounds self.data %d index %d"%(self.data.shape[0],self.t+2))
            else:
                self.t=self.t+1
                terminal=False
        '''
        TODO: modify above to handle when days are not continuous
        This requires to specify a feature that helps to keep track of 
        the day, for this specific dataset it could be the days to end of the semester variable
        (Done)
        '''
        return self.data[self.state_features].values[self.t,:],self.data[self.reward_feature].values[self.t,:][0],terminal,""
    
    def _reward_func(self, state, action):
        '''
        Args:
            state (Observation in the sequence)
            action (Observed action) : This is not used internally, however is kept for consistency
        
        Returns
            (float)
        '''
        
        obs, reward, is_terminal, info = self.step()

        if self.render:
            #Could be implemented in the future
            #To visualize progression over time
            pass

        self.next_state = DataState(obs, is_terminal=is_terminal)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (Observation in the sequence)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        pass
    #No need to make t=0
    #Preserved for consistency
#         self.t=0
    

    def __str__(self):
        return "data-" + str(self.env_name)

