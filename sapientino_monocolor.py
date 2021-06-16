import random
import time

from tensorforce import Agent, Environment
from tensorforce.environments import OpenAIGym

from gym.wrappers import TimeLimit

from gym_sapientino_case.env import SapientinoCase


class SapientinoCaseWrapper(Environment):

    def __init__(self):
        super().__init__()
        self.env = SapientinoCase(
          colors=["red"],
          map_file='monocolor_map.txt',
          logdir=".",
        )
        self.env = TimeLimit(self.env, 100)


    def states(self):
        return OpenAIGym.specs_from_gym_space(space=self.env.observation_space, allow_infinite_box_bounds=True)

    def actions(self):
        return OpenAIGym.specs_from_gym_space(space=self.env.action_space)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

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

    # Pre-defined or custom environment
    environment = Environment.create(
        environment=SapientinoCaseWrapper, max_episode_timesteps=100
    )
    
    # Instantiate a Tensorforce agent
    agent = Agent.create(
        agent='ac',
        environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
        batch_size=8,
#         memory=10000,
#         update=dict(unit='timesteps', batch_size=64),
#         optimizer=dict(type='adam', learning_rate=3e-4),
#         policy=dict(network='auto'),
#         objective='policy_gradient',
#         reward_estimation=dict(horizon=20)
    )
    
    episodes = 300
    render_interval = 5
    cum_rewards = []
        
    for ep in range(episodes):
        
        states = environment.reset()
        terminal = False
        cum_rewards.append(0.)
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            if reward > 0.:
                print('Goal reached')
            cum_rewards[ep] += reward
            
            agent.observe(terminal=terminal, reward=reward)

            if ep % render_interval == render_interval - 1:
                environment.render()        
                time.sleep(0.025)
        
        if ep % render_interval == render_interval - 1:
            avg_cum_reward = sum(cum_rewards[-render_interval:]) / render_interval
            print(f'Last {render_interval} episodes cum rewards:')
            print(cum_rewards[-render_interval:])
    
    agent.close()
    environment.close()
