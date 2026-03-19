from __future__ import annotations
import numpy as np 
class BitQueues: 
    """Real traffic queues Q_u(t). 
    Update rule: Q_u(t+1) = max(Q_u(t) - r_u(t), 0 ) + A_u(t)"""
    def __init__(self, num_ue: int, init_bits: float=0.0): 
        self.Q = np.full((num_ue, ), float(init_bits), dtype=np.float64) 
    
    def reset(self, init_bits: float=0.0) -> None: 
        self.Q.fill(float(init_bits)) 
        
    def update(self, service_bits: np.ndarray, arrival_bits: np.ndarray) -> None: 
        # Q(t + 1) = max(Q(t) - r(t), 0) + A(t) 
        self.Q = np.maximum(self.Q - service_bits, 0.0) + arrival_bits 


class VirtualQueues: 
    """Virtual queues for constrained DPP scheduling. 
    
    Z_u(t): enforces per-user minimum rate guarantee (F-C2) 
        Z_u(t + 1) = max(Z_u(t) - r_u(t), 0) + r_u^min 
    Y(t): enforces long-run average power budget    (F-C3) 
        Y(t+1) = max(Y(t) + P(t) - P^avg, 0) 
    These emerge directly from the Lyapunov drift expansion and are 
    NOT design choices
    """
    def __init__(self, num_ue: int): 
        self.Z = np.zeros(num_ue, dtype=np.float64) 
        self.Y: float = 0.0 
    
    def reset(self) -> None: 
        self.Z.fill(0.0) 
        self.Y = 0.0 

    def update_Z(self, service_bits: np.ndarray, r_min: float) -> None: 
        """Update per-user minimum-rate virtual queue, 
        
        Z_u(t+1) = max(Z_u(t) - r_u(t), 0) + r_u^min
        """
        self.Z = np.maximum(self.Z - service_bits, 0.0) + r_min 

    def update_Y(self, p_total: float, p_avg: float) -> None: 
        """Update power-budget virtual queue, 
        Y(t+1) = max(Y(t) + P(t) - P^avg, 0) 
        """
        self.Y = max(self.Y + p_total - p_avg, 0.0) 
    