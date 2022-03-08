import gym
import math
import os
from dotenv import load_dotenv
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

load_dotenv()

ACTIVATIONS = {
  'elu': nn.ELU,
  'hardshrink': nn.Hardshrink,
  #'hardsigmoid': nn.Hardsigmoid,
  'hardtanh': nn.Hardtanh,
  #'hardswish': nn.Hardswish,
  'leaky relu': nn.LeakyReLU,
  'log sigmoid': nn.LogSigmoid,
  'prelu': nn.PReLU,
  'relu': nn.ReLU,
  'relu 6': nn.ReLU6,
  'rrelu': nn.RReLU,
  'selu': nn.SELU,
  'celu': nn.CELU,
  'gelu': nn.GELU,
  'sigmoid': nn.Sigmoid,
  'silu': nn.SiLU,
  'mish': nn.Mish,
  'softplus': nn.Softplus,
  'softshrink': nn.Softshrink,
  'softsign': nn.Softsign,
  'tanh': nn.Tanh,
  'tanhshrinik': nn.Tanhshrink
}
CONSTANT_LR_CAP = float(os.environ['CONSTANT_LR_CAP'])
LR_TYPE = os.environ['LR_TYPE']
N_ENVS = int(os.environ['N_ENVS'])

activation_strs = ACTIVATIONS.keys()

class Algorithm():
  gamma = True
  net = True
  
  def __init__(self, env):
    self.env = env
    self.sde_available = isinstance(gym.make(env).action_space, gym.spaces.Box)
    
  def get_params(self):
    lr_type = self.suggest_categorical('learning rate type', ['constant', 'linear', 'exponential']) if LR_TYPE == 'auto' else LR_TYPE
    
    if lr_type == 'constant':
      lr = lambda _: self.suggest_float('constant lr value', 1e-5, CONSTANT_LR_CAP)
    elif lr_type == 'linear':
      lr = lambda prog_rem: prog_rem * self.suggest_float('linear lr coefficient', 1e-5, 0.5)
    else:
      lr = lambda prog_rem: self.suggest_float('exp lr coefficient', 1e-5, 0.5) * math.exp((prog_rem - 1) * self.suggest_float('exp lr decay', 1, 15))
    
    params = {
      'policy': 'MlpPolicy',
      'env': make_vec_env(self.env, n_envs=N_ENVS),
      'learning_rate': lr,
      'verbose': 0
    }
    
    if self.gamma:
      params['gamma'] = self.suggest_float('gamma', 0.9, 1)
    
    if self.net:
      params['policy_kwargs'] = {
        'activation_fn': ACTIVATIONS[self.suggest_categorical('activation fn', activation_strs)]
      }
    
    if self.sde_available:
      params['use_sde'] = self.suggest_categorical('use sde', [False, True])
      if params['use_sde']:
        resample_sde = self.suggest_categorical('resample sde', [False, True])
        params['sde_sample_freq'] = self.suggest_int('sample sde freq', 1, 256) if resample_sde else -1
        
    return params
    
  def gen_model(self, trial):
    self.trial = trial
    self.suggest_categorical = trial.suggest_categorical
    self.suggest_float = trial.suggest_float
    self.suggest_int = trial.suggest_int
    
    return self.alg(**self.get_params())
  
  def net(self, name):
    layers = self.suggest_int(name + ' layers', 0, 3)
    layer_size = self.suggest_int(name + ' layer size', 8, 512)
    return [layer_size for _ in range(layers)]
  
  def net_arch(self, name1, name2):
    net_arch = {}
    net_arch[name1] = self.net(name1)
    net_arch[name2] = self.net(name2)
    return net_arch
