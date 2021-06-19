import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
from nets import *
from gym_sapientino_case.env import SapientinoCase
import time
from gym.wrappers import TimeLimit



def state2tensor(state):
    modified_state = [state[0][0], state[0][1], state[0][2], state[0][3]]
    for el in state[1]:
        modified_state.append(el)
    state = np.array(modified_state)
    return torch.from_numpy(state).float()



class Memory():

    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs, \
                self.values, \
                self.rewards, \
                self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)


def train(memory, q_val, adam_critic, adam_actor, gamma=0.99):
    
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))

    # Target values are calculated backward
    # it's super important to handle correctly done states,
    # for those caes we want our to target to be equal to the reward only
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + gamma*q_val*(1.0-done)
        q_vals[len(memory)-1 - i] = q_val   # store values from the end to the beginning

    # Compute the advantage
    advantage = torch.Tensor(q_vals) - values
    
    # Compute the critic loss and update
    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()

    # Compute the actor loss and update
    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()



def main():

    # Argument parsing

    import sys
    import os
    import configparser
    import argparse

    parser = argparse.ArgumentParser(description='Train an AC agent on SapientinoCase.')
    parser.add_argument('experiment_dir')
    parser.add_argument('--render_interval', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=300)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    render_interval = args.render_interval
    episodes = args.episodes

    experiment_cfg = configparser.ConfigParser()
    experiment_cfg.read(os.path.join(experiment_dir, 'params.cfg'))
    env_cfg = experiment_cfg['ENVIRONMENT']
    agent_cfg = experiment_cfg['AGENT']

    colors = env_cfg['colors'].replace(' ', '').split(',')

    # Create the environment
    env = SapientinoCase(
        colors=colors,
        map_file=os.path.join(experiment_dir, env_cfg['map_file']),
        logdir=experiment_dir
    )

    # Set the max number of steps
    env = TimeLimit(env, env_cfg.getint("max_episode_timesteps"))

    # Initialize dimensions
    state_dim = 4 + 1
    n_actions = 5
    
    # Initialize the nets
    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)

    # Initialize optimizers
    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Initialize the memory
    memory = Memory()

    # MAIN LOOP
    batch_size = agent_cfg.getint('batch_size')
    n_episodes = episodes
    cum_rewards = []
    
    for ep in range(n_episodes):
        done = False
        total_reward = 0
        state = env.reset()
        steps = 0

        print(f"\rCurrent episode {ep}", end="")

        cum_rewards.append(0.)

        while not done:
            
            if ep % render_interval == render_interval - 1:
                env.render()
                time.sleep(0.01)

            # Sample the action from a Categorical distribution
            probs = actor(state2tensor(state))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            
            # Apply the action on the env and take the observation (next_state)
            next_state, reward, done, info = env.step(action.detach().data.numpy())
            
            cum_rewards[ep] += reward
            steps += 1
            state = next_state
            
            # Expand memeory
            memory.add(dist.log_prob(action), critic(state2tensor(state)), reward, done)

            # Train if done or num steps > batch_size
            if done or (steps % batch_size == 0):
                last_q_val = critic(state2tensor(next_state)).detach().data.numpy()
                train(memory, last_q_val, adam_critic, adam_actor)
                memory.clear()
            
        if ep % 10 == 9:
            avg_cum_reward = sum(cum_rewards[-10:]) / 10
            print(f'\nLast 10 episodes avg cum rewards: ', end="")
            print(sum(cum_rewards[-10:]) / 10)


if __name__ == '__main__':
    main()
