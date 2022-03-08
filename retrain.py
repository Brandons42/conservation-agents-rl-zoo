import numpy as np
import gym
import gym_fishing
import gym_climate
import gym_conservation
from sb3_contrib import ARS, QRDQN, TQC, TRPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
from optuna import visualization
import argparse
import os
from torch import nn
from stable_baselines3.common.monitor import Monitor

evaluate = lambda model: print(evaluate_policy(model, Monitor(gym.make('conservation-v6')), n_eval_episodes=100))

model = TQC.load('conservation-v6/tqc/2022-02-2108:47:37.098317/model-168.0982022.zip')
#params = original_model.get_parameters()
#print(params['policy'].keys())
#model = A2C(**params)
#model.learn(total_timesteps=10000)
evaluate(model)

#vec_env = make_vec_env('fishing-v1', n_envs=4)
#model.set_env(vec_env)
#model.learn(total_timesteps=1000000)

#evaluate(model)

#print([env.get_episode_rewards() for env in vec_env.envs])
