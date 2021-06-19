import random
import time

from tensorforce import Agent, Environment
from tensorforce.environments import OpenAIGym

from gym.wrappers import TimeLimit

from gym_sapientino_case.env import SapientinoCase



if __name__ == '__main__':

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
   
    environment = Environment.create(
        environment='gym', 
        level='SapientinoCase-v0', 
        max_episode_timesteps=env_cfg.getint("max_episode_timesteps"),
        colors=colors,
        map_file=os.path.join(experiment_dir, env_cfg['map_file']),
        logdir=experiment_dir
    )

    agent = Agent.create(
        agent='a2c',
        network=os.path.join(experiment_dir, agent_cfg['policy_network']),
        critic=os.path.join(experiment_dir, agent_cfg['critic_network']),
        environment=environment,
        batch_size=agent_cfg.getint('batch_size'),
    )

    cum_rewards = []
    
    print(agent.get_architecture())
        
    for ep in range(episodes):
    
    	# Record episode experience
        episode_states = list()
        episode_internals = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()
        
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        cum_rewards.append(0.)
        
        while not terminal:
            episode_internals.append(internals)
            episode_states.append(states)

            actions, internals = agent.act(states=states, internals=internals, independent=True)
            
            episode_actions.append(actions)
            
            states, terminal, reward = environment.execute(actions=actions)
            
            episode_terminal.append(terminal)
            episode_reward.append(reward)
            
            cum_rewards[ep] += reward    
            
                    
        if ep % render_interval == render_interval - 1:
            avg_cum_reward = sum(cum_rewards[-render_interval:]) / render_interval
            print(f'Last {render_interval} episodes avg cum rewards:')
            print(sum(cum_rewards[-render_interval:]) / render_interval)
        
        # Feed recorded experience to agent
        agent.experience(
            states=episode_states, internals=episode_internals, actions=episode_actions,
            terminal=episode_terminal, reward=episode_reward
        )

        # Perform update
        agent.update()
    
    agent.close()
    environment.close()
