from sb3_contrib import QRDQN as QRDQN_alg, TQC as TQC_alg
from stable_baselines3 import DDPG as DDPG_alg, DQN as DQN_alg, SAC as SAC_alg, TD3 as TD3_alg

from .Algorithm import Algorithm

class OffPolicy(Algorithm):
  def get_params(self):
    params = {
      **Algorithm.get_params(self),
      'buffer_size': self.suggest_int('buffer size', 1e4, 1e6),
      'learning_starts': self.suggest_int('learning starts', 0, 100000),
      'batch_size': self.suggest_int('batch size', 16, 2048),
      'tau': self.suggest_float('tau', 0.001, 0.1),
      #'train_freq': (
      #  self.suggest_int('train freq num', 1, 512),
      #  self.suggest_categorical('train freq type', ['step', 'episode'])
      #),
      'train_freq': self.suggest_int('train freq', 1, 512),
      'gradient_steps': self.suggest_int('gradient steps', 1, 512),
    }
  
    params['policy_kwargs']['net_arch'] = self.net_arch('pi', 'qf')
    
    if self.sde_available and params['use_sde']:
      params['use_sde_at_warmup'] = self.suggest_categorical('use sde at warmup', [False, True])
      
    return params

class DDPG(OffPolicy):
  alg = DDPG_alg
  alg_str = 'ddpg'
  
  def __init__(self, env):
    super().__init__(env)
    
    self.sde_available = False

class DQN(OffPolicy):
  alg = DQN_alg
  alg_str = 'dqn'
  
  def __init__(self, env):
    super().__init__(env)
    
    self.sde_available = False
  
  def get_params(self):
    return {
      **OffPolicy.get_params(self),
      #'env': gym.make(args.env),
      'target_update_interval': self.suggest_int('target update interval', 1, 20000),
      'exploration_fraction': self.suggest_float('exploration fraction', 0, 0.5),
      'exploration_initial_eps': self.suggest_float('exploration initial eps', 0.8, 1),
      'exploration_final_eps': self.suggest_float('exploration final eps', 0, 0.1),
      #'max_grad_norm': self.suggest_float('max grad norm')
    }

class QRDQN(DQN):
  alg = QRDQN_alg
  alg_str = 'qr-dqn'
  
  def get_params(self):
    params = DQN.get_params(self)
    
    params['policy_kwargs']['n_quantiles'] = self.suggest_int('n quantiles', 5, 50)
    
    return params

class SAC(OffPolicy):
  alg = SAC_alg
  alg_str = 'sac'

class TD3(OffPolicy):
  alg = TD3_alg
  alg_str = 'td3'

  def __init__(self, env):
    super().__init__(env)
    
    self.sde_available = False

  def get_params(self):
    return {
      **OffPolicy.get_params(self),
      #'policy_delay': self.suggest_int('policy delay'),
      #'target_policy_noise': self.suggest_float('target policy noise'),
      #'target_policy_clip': self.suggest_float('target policy clip')
    }

class TQC(OffPolicy):
  alg = TQC_alg
  alg_str = 'tqc'
  
  def get_params(self):
    n_quantiles = self.suggest_int('n quantiles', 5, 50)
    
    params = {
      **OffPolicy.get_params(self),
      'top_quantiles_to_drop_per_net': self.suggest_int("top_quantiles_to_drop_per_net", 0, n_quantiles - 1)
    }
    
    #params['policy_kwargs']['n_critics']
    params['policy_kwargs']['n_quantiles'] = n_quantiles
    
    return params
