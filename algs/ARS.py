from sb3_contrib import ARS as ARS_alg

from .Algorithm import Algorithm

class ARS(Algorithm):
  alg = ARS_alg
  alg_str = 'ars'
  gamma = False
  net = False
  
  def __init__(self, env):
    super().__init__(env)
    
    self.sde_available = False
    
  def get_params(self):
    return{
      **Algorithm.get_params(self),
      'delta_std': self.suggest_float('delta std', 0.01, 0.3),
      'n_delta': self.suggest_int('n delta', 4, 64),
      'policy': 'LinearPolicy' if self.suggest_categorical('lin policy', [False, True]) else 'MlpPolicy',
      'zero_policy': self.suggest_categorical('zero policy', [False, True])
    }
