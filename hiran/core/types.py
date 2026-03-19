# hiran/core/types.py 

from __future__ import annotations 
from dataclasses import dataclass, field  
from typing import Dict, Any, Optional, List 
import numpy as np 

@dataclass 
class EnvConfig: 
    # -- Dimensions -- 
    num_bs: int = 3 
    num_ue: int = 20 
    num_prb: int = 25 

    # -- Two-time-scale -- 
    W: int = 200        # Slots per window  
    K: int = 50         # windows per episode 

    # -- Traffic / Queue -- 
    arrival_rate_bits: float = 2e5      # mean bits / slot per UE (Poisson) 
    queue_init_bits: float = 0.0        

    # -- Radio -- 
    prb_bw_hz: float = 180e3    
    noise_power_w: float = 1e-10 
    p_bs_max_w: float = 10.0            # max TX power (simplified) 
    pathloss_exp: float = 3.5 
    shadowing_std_db: float = 6.0 
    enable_shadowing: bool = True 

    # -- DPP Control parameter -- 
    V: float = 0.0 
    V_norm: float = 1e9                 # Normalization: V_eff = V * V_norm + Y 

    # -- Energy model (Bjornson two-component) -- 
    P0_bs: float=50.0                   # static circuit power (W) - broadcasts to all BSs 
    delta_P_bs: float=30.0              # dynamic power slope - broadcasts to all BSs 
    p_sleep_w: float=5.0                # sleep-mode power (W) 

    # Deprecated but kept so old scripts don't break 
    p_on_w: float=50.0 
    load_power_coeff: float=0.0 

    # -- Constrained DPP parameters -- 
    r_min_bits: float=1e4               # per-user minimum rate r_u^min (bits/slot) 
    P_avg: float=120.0                  # long-run average power budget P^avg (W) 

    # -- Reward weights --
    eps_pf: float = 1e-9 
    eta_energy: float = 1e-4 
    eta_switch: float = 0.1 
    eta_queue: float = 1e-6 

    # Bias action range
    bias_min_db: float = -6.0 
    bias_max_db: float = 6.0 

    # -- Bias smoothness constraint -- 
    delta_bias_max: float = 3.0         # max per-window bias change |beta(k+1) - beta(k)| <= delta_bias_max 

    # -- Topology box (meters) -- 
    area_size_m: float = 500.0 

    # -- Logging -- 
    log_dir: str = 'logs' 
    enable_jsonl_logger: bool = True 
    enable_slot_trace: bool = False     # per-slot trace (expensive, off by default) 

    # Misc 
    seed: int = 0


@dataclass 
class KPIs: 
    avg_thr_ue: np.ndarray                 # (U, )     window-avg throughput per UE 
    avg_q_ue: np.ndarray                    # (U, )     window-avg real queue per UE 
    util_bs: np.ndarray                     # (B, )     window-avg utilization per BS 
    energy_window: float                    # total energy over the window (W·slots) 
    switch_count: int                       
    # constrained DPP diagnostics 
    avg_Z_ue: np.ndarray                    # (U, )     window-avg virtual queue Z per UE 
    avg_Y: float                            # window-avg virtual queue Y  
    avg_power_per_bs: np.ndarray            # (B, )     window-avg power per BS 
    prbs_blanked_frac: float                # fraction of total PRB-slots left blank 
    extra: Dict[str, any] = field(default=dict) 


@dataclass 
class StepResult: 
    obs: Dict[str, np.ndarray] 
    reward: float 
    done: bool 
    info: Dict[str, Any] 


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray: 
    return np.minimum(np.maximum(x, lo), hi) 
