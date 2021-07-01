import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import animation
from gym_sapientino_case.env import SapientinoCase
import time
from gym.wrappers import TimeLimit
import importlib

import sys
import os
import configparser
import argparse
import pickle



# A function that format the state given by the
# sapientino environment in such a way it is
# good to be putted in the nets
def state2tensor(state):
    modified_state = [state[0][0], state[0][1], state[0][2], state[0][3]]
    for el in state[1]:
        modified_state.append(el)
    state = np.array(modified_state)
    return torch.from_numpy(state).float()



def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Plot stuff.')
    parser.add_argument('experiment_dir')
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    history_file = os.path.join(experiment_dir, "history.pkl")

    # Import the networks from the folder
    if(experiment_dir[-1] == "/"):
        nets_dir = experiment_dir[:-1]
    else:
        nets_dir = experiment_dir
    Actor = getattr(importlib.import_module(nets_dir + ".nets"), "Actor")
    Critic = getattr(importlib.import_module(nets_dir + ".nets"), "Critic")

    experiment_cfg = configparser.ConfigParser()
    experiment_cfg.read(os.path.join(experiment_dir, 'params.cfg'))
    env_cfg = experiment_cfg['ENVIRONMENT']
    agent_cfg = experiment_cfg['AGENT']
    other_cfg = experiment_cfg['OTHER']

    colors = env_cfg['colors'].replace(' ', '').split(',')
    map_file = os.path.join(experiment_dir, env_cfg['map_file'])
    max_episode_timesteps = env_cfg.getint("max_episode_timesteps")

    # Create the environment
    env = SapientinoCase(
        colors=colors,
        map_file=map_file,
        logdir=experiment_dir
    )

    # Set the max number of steps
    env = TimeLimit(env, max_episode_timesteps)

    # Initialize dimensions
    state_dim = 4 + 1
    n_actions = 5
    n_states = env.observation_space[-1].nvec[0]

    # Initialize the nets
    actor = Actor(state_dim, n_actions, n_states)
    critic = Critic(state_dim, n_states)

    actor.load_model_weights(os.path.join(experiment_dir, "actor.weights"))
    critic.load_model_weights(os.path.join(experiment_dir, "critic.weights"))

    with open(history_file, "rb") as f:
        cum_rewards, steps, time = pickle.load(f)

    episodes = len(cum_rewards)
    print(f'Plotting {episodes} episodes')
    gamma = 0.99
    discounted_rewards = [reward*(gamma**n_steps) for reward, n_steps in zip(cum_rewards, steps)]
    
    plt.scatter(range(len(discounted_rewards)), discounted_rewards, color='blue', marker='.')
    plt.show()

    sapientino_eval(actor, critic, env, experiment_dir, n_episodes=1)

    

def sapientino_eval(actor, critic, env, experiment_dir, n_episodes=1):

    # Eval
    print("Evaluation Started.")
    tot_reward = 0.0
    ep = 0
    for ep in range(n_episodes):
        state = env.reset()
        done = False

        print(f"\rCurrent episode: {ep}", end="")

        frames = []

        while not done:
            
            frames.append(env.render(mode="rgb_array"))
            time.sleep(0.05)
            
            # Sample the action from a Categorical distribution
            probs = actor(state2tensor(state))
            action = torch.argmax(probs, dim=0)
            action = action.detach().data.numpy()

            # Apply the action on the env and take the observation (next_state)
            next_state, reward, done, info = env.step(action)
            
            # Udpate the state
            state = next_state

            tot_reward += reward
    
    print(f"\nEVAL RESULT: {tot_reward}")
    
    save_frames_as_gif(frames, path=experiment_dir, filename='eval.gif')


"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)



if __name__ == '__main__':
    main()
