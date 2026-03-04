from __future__ import annotations
import numpy as np 

class BitQueues: 
    def __init__(self, num_ue: int, init_bits: float=0.0): 
        self.Q = np.full((num_ue, ), float(init_bits), dtype=np.float64) 
    
    def reset(self, init_bits: float=0.0) -> None: 
        self.Q.fill(float(init_bits)) 

    def update(self, service_bits: np.ndarray, arrival_bits: np.ndarray) -> None: 
        # Q(t + 1) = max(Q(t) - r(t), 0) + A(t) 
        self.Q = np.maximum(self.Q - service_bits, 0.0) + arrival_bits 
