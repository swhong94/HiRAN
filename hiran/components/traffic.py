# hiran/componenents/traffic.py 

from __future__ import annotations
import numpy as np


class PoissonTraffic: 
    def __init__(self, arrival_rate_bits: float, rng: np.random.Generator): 
        self.lam = float(arrival_rate_bits) 
        self.rng = rng 
    
    def sample(self, num_ue: int) -> np.ndarray: 
        # Poisson in bits/slot (integer) 
        return self.rng.poisson(lam=self.lam, size=(num_ue,)).astype(np.float64) 