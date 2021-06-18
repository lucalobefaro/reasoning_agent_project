import random
import time

from tensorforce import Agent, Environment
from tensorforce.environments import OpenAIGym

from gym.wrappers import TimeLimit

from gym_sapientino_case.env import SapientinoCase


class SapientinoCaseWrapper(Environment):

    def __init__(self, colors, map_file, logdir='.'):
        super().__init__()
        self.env = SapientinoCase(
          colors=colors,
          map_file=map_file,
          logdir=logdir,
        )
        self.env = TimeLimit(self.env, 100)

    def states(self):
        return OpenAIGym.specs_from_gym_space(space=self.env.observation_space, allow_infinite_box_bounds=True)

    def actions(self):
        return OpenAIGym.specs_from_gym_space(space=self.env.action_space)

    def reset(self):
        states = self.env.reset()
        states = OpenAIGym.flatten_state(state=states, states_spec=self.states())
        return states

    def execute(self, actions):
        next_state, reward, terminal, _ = self.env.step(actions)
        if reward == 1.:
            terminal = True
        return OpenAIGym.flatten_state(state=next_state, states_spec=self.states()), terminal, reward

    def render(self):
        self.env.render()


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
        max_episode_timesteps=100,
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
        
        states = environment.reset()
        terminal = False
        cum_rewards.append(0.)
        
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            cum_rewards[ep] += reward    
            agent.observe(terminal=terminal, reward=reward)
            if ep % render_interval == render_interval - 1:
                environment.environment.render()        
                time.sleep(0.025)
        
        if ep % render_interval == render_interval - 1:
            avg_cum_reward = sum(cum_rewards[-render_interval:]) / render_interval
            print(f'Last {render_interval} episodes avg cum rewards:')
            print(sum(cum_rewards[-render_interval:]) / render_interval)
    
    agent.close()
    environment.close()
