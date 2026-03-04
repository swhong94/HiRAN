# hiran/core/types.py 

from __future__ import annotations 
from dataclasses import dataclass 
from typing import Dict, Any, Optional 
import numpy as np 

@dataclass 
class EnvConfig: 
    # Dimensions: 
    num_bs: int = 3 
    num_ue: int = 20 
    num_prb: int = 25 

    # Two-time-scale 
    W: int = 200        # Slots per window  
    K: int = 50         # windows per episode 

    # Traffic / Queue 
    arrival_rate_bits: float = 2e5      # mean bits / slot per UE (Poisson) 
    queue_init_bits: float = 0.0        

    # Radio 
    prb_bw_hz: float = 180e3    
    noise_power_w: float = 1e-10 
    p_bs_max_w: float = 10.0            # max TX power (simplified) 
    pathloss_exp: float = 3.5 
    shadowing_std_db: float = 6.0 
    enable_shadowing: bool = True 

    # DPP 
    V: float = 0.0 

    # Energy model 
    p_on_w: float = 50.0 
    p_sleep_w: float = 5.0 
    load_power_coeff: float = 0.0       # set >0 later if you want load-dependent energy 

    # Reward weights 
    eps_pf: float = 1e-9 
    eta_energy: float = 1e-4 
    eta_switch: float = 0.1 

    # Bias action range
    bias_min_db: float = -6.0 
    bias_max_db: float = 6.0 

    # Topology box (meters) 
    area_size_m: float = 500.0 

    # Misc 
    seed: int = 0


@dataclass 
class KPIs: 
    avg_thr_ue: np.ndarray      # (U, ) 
    avg_q_ue: np.ndarray        # (U, ) 
    util_bs: np.ndarray         # (B, ) 
    energy_window: float        
    switch_count: int 
    extra: Dict[str, Any] 


@dataclass 
class StepResult: 
    obs: Dict[str, np.ndarray] 
    reward: float 
    done: bool 
    info: Dict[str, Any] 


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray: 
    return np.minimum(np.maximum(x, lo), hi) 
