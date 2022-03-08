import os
from dotenv import load_dotenv
from sb3_contrib import TRPO as TRPO_alg
from stable_baselines3 import A2C as A2C_alg, PPO as PPO_alg

from .Algorithm import Algorithm

load_dotenv()

N_ENVS = int(os.environ['N_ENVS'])

class OnPolicy(Algorithm):
  batch_size = False
  contrib = False
  
  def get_params(self):
    params = {
      **Algorithm.get_params(self),
      'n_steps': 2 ** self.suggest_int('log n steps', 4, 13),
      'gae_lambda': self.suggest_float('gae lambda', 0.8, 1),
      #'ent_coef': self.suggest_float('ent coef', 0, 0.2),
    }
    
    if not self.contrib:
      params['vf_coef'] = self.suggest_float('vf coef', 0, 1)
      params['max_grad_norm'] = self.suggest_float('max grad norm', 0.3, 5)
  
    if self.batch_size:
      params['batch_size'] = params['n_steps'] * N_ENVS // self.suggest_categorical('batch size divider', [1, 2, 4])
  
    shared_layer_size = self.suggest_int('shared layer size', 8, 512)
    params['policy_kwargs']['net_arch'] = [shared_layer_size, self.net_arch('vf', 'pi')]
    params['policy_kwargs']['ortho_init'] = self.suggest_categorical('ortho init', [False, True])
    
    return params

class A2C(OnPolicy):
  alg = A2C_alg
  alg_str = 'a2c'
    
  def get_params(self):
    params = {
      **OnPolicy.get_params(self),
      'use_rms_prop': self.suggest_categorical('use rms prop', [False, True]),
      'normalize_advantage': self.suggest_categorical('normalize advantage', [False, True])
    }
  
    if params['use_rms_prop']:
      params['rms_prop_eps'] = self.suggest_float('rms prop eps', 0, 0.1)
      
    return params

class PPO(OnPolicy):
  alg = PPO_alg
  alg_str = 'ppo'
  batch_size = True
  
  def get_params(self):
    params = OnPolicy.get_params(self)
    
    return {
      **params,
      'n_epochs': self.suggest_int('n epochs', 1, 30),
      'clip_range': self.suggest_float('clip range', 0, 0.5)
    }

class TRPO(OnPolicy):
  alg = TRPO_alg
  alg_str = 'trpo'
  batch_size = True
  contrib = True
  
  def get_params(self):
    return {
      **OnPolicy.get_params(self),
      'cg_max_steps': self.suggest_int('cg max steps', 5, 30),
      'cg_damping': self.suggest_float('cg damping', 0.01, 0.5),
      'line_search_shrinking_factor': self.suggest_float('line search shrinking factor', 0.6, 0.9),
      'n_critic_updates': self.suggest_int('n critic updates', 5, 30),
      'target_kl': self.suggest_float('target kl"', 0.001, 0.1),
      'normalize_advantage': self.suggest_categorical('normalize advantage', [False, True])
    }
