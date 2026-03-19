# hiran/components/energy.py 

from __future__ import annotations
import numpy as np 

class LoadDependentEnergyModel: 
    """Following the Björnson et al. two-component power model
    
    Per-BS per-slot power: 
        P_b(t) = s_{b, k} (P0 + delta_P * util(t) + (1 - s_{b, k}) * P_sleep)
    
    where: 
        P0 := P_fix^b + M_b * P_tc^b       (static circuit power) 
        P^b := Delta_b^sp + p_b / eta_b     (dynamic power slope)

    The model is independent of the channel model - valid under both 
    SNR-only and SINR with interference 

    Args: 
        P0_per_bs:      (B, )   static circuit power per BS             [W]
        delta_P_per_bs: (B, )   dynamic power slope per BS              [W]
        p_sleep_w:      sleep_mode_power (scaler, same for all BSs)     [W]    
    """
    def __init__(self, 
                 P0_per_bs: np.ndarray, 
                 delta_P_per_bs: np.ndarray, 
                 p_sleep_w):
        self.P0 = np.asarray(P0_per_bs, dtype=np.float64)               # (B, ) 
        self.delta_P = np.asarray(delta_P_per_bs, dtype=np.float64)     # (B, ) 
        self.p_sleep = float(p_sleep_w) 

    def power_slot(self, 
                   bs_on: np.ndarray, 
                   util_bs: np.ndarray) -> tuple[float, np.ndarray]: 
        """Compute per-BS and total power for one slot. 
        
        Args: 
            bs_on: (B, ) ON/OFF np.ndarray 
            util_bs: (B, ) fraction of PRBs actually used
            
        Returns: 
            p_total:    scalar total power across all BSs   [W] 
            p_per_bs:   (B, ) per-BS power                  [W] 
        """
        on = (bs_on > 0.5).astype(np.float64) 
        p_per_bs = (on * (self.P0 + self.delta_P * util_bs) + (1.0 - on) * self.p_sleep) 
        return float(np.sum(p_per_bs)), p_per_bs 

