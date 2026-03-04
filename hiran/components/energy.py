from __future__ import annotations
import numpy as np 

class SimpleEnergyModel: 
    def __init__(self, p_on_w: float, p_sleep_w: float, load_coeff: float=0.0): 
        self.p_on = float(p_on_w) 
        self.p_sleep = float(p_sleep_w) 
        self.load_coeff = float(load_coeff) 
    
    def power_slot(self, bs_on: np.ndarray, util_bs: np.ndarray) -> float: 
        # Power = sum_b [on * (P_on + load_coeff *util) + (1-on) * P_sleep] 
        on = (bs_on > 0.5).astype(np.float64) 
        return float(np.sum(on * (self.p_on + self.load_coeff * util_bs) + (1 - on) * self.p_sleep))
    