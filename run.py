import argparse
import gym
import gym_fishing
import gym_climate
import gym_conservation
import numpy as np
import optuna
import os
from datetime import datetime
from dotenv import load_dotenv
from optuna import visualization
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList, EveryNTimesteps
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from algs.ARS import ARS
from algs.OffPolicy import *
from algs.OnPolicy import *

load_dotenv()

ALGS = [A2C, ARS, DDPG, DQN, PPO, QRDQN, SAC, TD3, TQC, TRPO]
CB_INTERVAL = int(os.environ['CB_INTERVAL'])
DEFAULT_TRIALS = int(os.environ['DEFAULT_TRIALS'])
EVAL_EPS = int(os.environ['EVAL_EPS'])
N_ENVS = int(os.environ['N_ENVS'])
TRAIN_STEPS = int(os.environ['TRAIN_STEPS'])

parser = argparse.ArgumentParser('RL Leaderboard Submission Generator')
parser.add_argument('env', help='Environment to study', type=str)
parser.add_argument('--lib', default='sb3', help='RL library to use (keras, sb3 (default))')
parser.add_argument('--alg', default='ppo', help='Algorithm to use (a2c, ddpg, dqn, ppo (default), sac, td3)', type=str)
parser.add_argument('--trials', default=DEFAULT_TRIALS, help='Number of hyperparameter tuning trials to run', type=int)
args = parser.parse_args()

class ReportCallback(BaseCallback):
  def __init__(self, trial, eval_callback):
    super(ReportCallback, self).__init__()
    self.trial = trial
    self.eval_callback = eval_callback
    
  def _on_step(self):
    #self.trial.report(self.eval_callback.last_mean_reward, step=self.parent.n_calls)
    self.trial.report(self.eval_callback.best_mean_reward, step=self.parent.n_calls)
    if self.trial.should_prune() or (args.env == 'fishing-v1' and self.eval_callback.last_mean_reward < 0.7):
      raise optuna.TrialPruned()
    return True

def choose_algorithm(alg_str):
  alg_str = alg_str.lower()
  chosen_alg = PPO
  
  for alg in ALGS:
    if alg.alg_str == alg_str:
      chosen_alg = alg
      break
    
  print('Using algorithm: ', chosen_alg.alg_str)
  return chosen_alg

alg = choose_algorithm(args.alg)(args.env)

best_model = None
best = float('-inf')

if not os.path.exists('./' + args.env):
  os.mkdir(args.env)
  
if not os.path.exists('./' + args.env + '/' + alg.alg_str):
  os.mkdir(args.env + '/' + alg.alg_str)

loc = args.env + '/' +  alg.alg_str + '/' + str(datetime.now()).replace(' ', '')

if not os.path.exists('./' + loc):
  os.mkdir(loc)

path = loc + '/'

def objective(trial):
  tmp = path + 'tmp/'
  
  model = alg.gen_model(trial)
  base_env = Monitor(gym.make(args.env))
  eval_callback = EvalCallback(base_env, best_model_save_path=tmp, eval_freq=CB_INTERVAL, n_eval_episodes=EVAL_EPS)
  report_callback = EveryNTimesteps(CB_INTERVAL * N_ENVS, ReportCallback(trial, eval_callback))
  
  model.learn(total_timesteps=TRAIN_STEPS, callback=CallbackList([eval_callback, report_callback]))
  mean_reward = eval_callback.best_mean_reward
  
  global best_model, best
  
  if mean_reward > best:
    best_model = alg.alg.load(tmp + 'best_model.zip')
    best = mean_reward
  
  return mean_reward

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, n_trials=args.trials, catch=(RuntimeError, ValueError))

best_model.save(path + 'model-' + str(study.best_trial.value) + '.zip')
best_params = study.best_params
print(best_params)

write = lambda viz, name: viz.write_image(path + name + '.png')

write(visualization.plot_slice(study), 'slice')
write(visualization.plot_intermediate_values(study), 'intermediate_values')
write(visualization.plot_optimization_history(study), 'optimization_history')
lr_params = ['learning rate type', 'constant lr value', 'linear lr coefficient', 'exp lr coefficient', 'exp lr decay']
write(visualization.plot_contour(study, params=list(filter(lambda param: param in study.best_params, lr_params))), 'contour')
#viz(visualization.plot_parallel_coordinate, 'parallel_coordinate')
#viz(visualization.plot_param_importances, 'param_importances')

env = gym.make(args.env)
if hasattr(env, 'simulate') and hasattr(env, 'plot'):
  df = env.simulate(best_model, reps=10)
  env.plot(df, path + 'sim.png')
if hasattr(env, 'policyfn') and hasattr(env, 'plot_policy'):
  pol_df = env.policyfn(best_model, reps=10)
  env.plot_policy(pol_df, path + 'policy.png')
