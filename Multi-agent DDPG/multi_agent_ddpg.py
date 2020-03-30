# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:04:25 2020

@author: elif.ayvali
"""
from ddpg_agent import DDPGAgent

import torch
import numpy as np
import random
from collections import namedtuple, deque
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Parameters from OpenAI Baselines
BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 256         # minibatch size
UPDATE_EVERY = 3         # Number of episodes that should elapse between gradient descent updates
GAMMA = 0.995            # discount factor
TAU = 1e-3               # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC =5e-4        # learning rate of the critic
WEIGHT_DECAY = 0         # L2 weight decay
EPSILON = 1.0            # Exploration noise coefficient
EPSILON_DECAY = 0        # Decay rate for exploration noise
LEARN_TIMES= 2           # Number of times to backprop with the batch
WARM_UP=   0             # Number of steps for uniform-random action selection, before running real policy. Helps exploration.
CLIP_NORM=True




class MADDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed):
        super(MADDPG, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.random_seed=random_seed
        self.num_agents=num_agents
        self.episode_num=0
        self.total_steps=0

        # Initialize centralized memory buffer
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # Initialize decentralized actors/critics
        self.maddpg_agents=self.get_agents()       
    

    def get_agents(self):
        """get all the agents in the MADDPG object"""
        agents = [DDPGAgent(self,idx) for idx in range(self.num_agents)]
        return agents
    
    
    def reset_agents(self):
        for agent in self.maddpg_agents:
            agent.reset()
            
    def act(self, states,  add_noise=True):
        """get actions from all agents in the MADDPG object"""
        actions=[agent.act(state,self.episode_num, add_noise) for agent, state in zip(self.maddpg_agents, states)] 
        return actions

    def encode_buffer(self,obs):
        #Concatenate states/actions of all agents
        #Reference : github@fsasilva59
#        print('before encode,obs',obs)
#        print('after encode,obs', np.array(obs).reshape(1,-1).squeeze())
        return np.array(obs).reshape(1,-1).squeeze()
           
    def decode_buffer(self,size, agent_idx, obs):
        #Reference : github@fsasilva59
        list_idx  = torch.tensor([ idx for idx in range(agent_idx * size, agent_idx * size + size) ]).to(device)    
#        print('before decode,obs',obs)
#        print('after decode,obs', obs.index_select(1, list_idx))
        return obs.index_select(1, list_idx)
        
    def step(self, states, actions, rewards, next_states, dones,episode):
        """Concatanate experience of all agents  in replay memory, and use random sample from buffer to learn."""
        self.episode_num=episode        
        self.memory.add(self.encode_buffer(states), self.encode_buffer(actions), self.encode_buffer(rewards), self.encode_buffer(next_states), self.encode_buffer(dones))     
        # Learn periodically, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.episode_num % UPDATE_EVERY==0:
            for _ in range(LEARN_TIMES):
                for idx in range(self.num_agents):   
                    experiences = self.memory.sample()    
                    self.learn(experiences,self.maddpg_agents,idx, GAMMA)                 
        self.total_steps+=1
                
    def save_checkpoint(self):
        for agent in self.maddpg_agents:
            torch.save(agent.actor_local.state_dict(), 'agent_'+str(agent.agent_idx)+'_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'agent_'+str(agent.agent_idx)+'_checkpoint_critic.pth')

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def learn(self,  experiences,meta_agent, agent_idx, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #all experiences
        #states: batch_sizex56
        #actions:batch_sizex4
        #rewards, batch_sizex2
        states, actions, rewards, next_states, dones = experiences
#        print('states',states)
#        print('actions',actions)
#        print('rewards',rewards)
        #get agent's experiences   
        agent=meta_agent[agent_idx]
#        print('agent object',agent)
        agent_states =  self.decode_buffer(self.state_size, agent_idx, states)
        agent_actions = self.decode_buffer(self.action_size, agent_idx, actions)
        agent_next_states = self.decode_buffer(self.state_size, agent_idx, next_states) 
        agent_rewards=self.decode_buffer(1, agent_idx, rewards) 
        agent_dones=self.decode_buffer(1, agent_idx, dones) 
#        print('agent_rewards',agent_rewards)
#        print('agent_dones',agent_dones)
#        print('agent_states',agent_states)

        #get other agents' experiences
        other_agent_idx=np.delete(range(self.num_agents),agent_idx).squeeze()
        other_agent=meta_agent[other_agent_idx]
#        print('other_agent object',other_agent)
        other_agent_states =  self.decode_buffer(self.state_size, other_agent_idx, states)
        other_agent_actions = self.decode_buffer(self.action_size, other_agent_idx, actions)
        other_agent_next_states = self.decode_buffer(self.state_size, other_agent_idx, next_states) 
        other_agent_rewards=self.decode_buffer(1, other_agent_idx, rewards) 
        #All agents ->torch for centralized critic
        all_states=torch.cat((agent_states, other_agent_states), dim=1).to(device)
        all_actions=torch.cat((agent_actions, other_agent_actions), dim=1).to(device)
        all_next_states=torch.cat((agent_next_states, other_agent_next_states), dim=1).to(device)
              
#        print('other agent_states',other_agent_states)

        # ---------------------------- update critic ---------------------------- #  
        # Get predicted next-state actions and Q values from target model
        agent_next_actions=agent.actor_target(agent_states)
        other_agent_next_actions=other_agent.actor_target(other_agent_states) #should be agent 2
#        print(' agent_nex_actions',agent_next_actions)
#        print('other_agent_next_actions',other_agent_next_actions)

        #Next actions-> torch
        all_next_actions=torch.cat((agent_next_actions,other_agent_next_actions), dim=1).to(device) 
   
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions) #batch_sizex1
#        print('Q_targets_next',Q_targets_next)
        # Compute Q targets for current states (y_i)
        
        Q_targets = agent_rewards + (gamma * Q_targets_next * (1 - agent_dones))
        
        # Compute critic loss
        Q_expected = agent.critic_local(all_states, all_actions)  #batch_sizex1
#        print('Q_expected',Q_expected)

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if CLIP_NORM is True:
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1) # clip gradient to max 1
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute next action predictions for all agents        
        agent_action_predictions=agent.actor_local(agent_states)
        #Predictions-> torch, Only backprop agent idx, detach other agents
        other_agent_action_predictions=other_agent.actor_local(other_agent_states).detach()
        all_actions_pred = torch.cat((agent_action_predictions, other_agent_action_predictions), dim = 1).to(device)        
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()

        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(agent.critic_local, agent.critic_target, TAU)
        self.soft_update(agent.actor_local, agent.actor_target, TAU)    

        # ---------------------------- update noise ---------------------------- #
        agent.epsilon -= EPSILON_DECAY
        agent.noise.reset()
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self,action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
#        print('Replay seed:',seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
#        print('new_experience :state',state)
#        print('new_experience :reward',reward)

        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences     =   random.sample(self.memory, k=self.batch_size)
   
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
