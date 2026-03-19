# !/usr/bin/env python3 
"""V-tradeoff with normalization - should see clear O(V) / O(1/V) behavior"""

import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np 
from hiran.core.types import EnvConfig
from hiran.core.env import TwoTimeScaleEnv


print("=== V-tradeoff with V-norm=1e9 ===\n") 
print(f"    {'V':>8s} | {'energy/w':>10s} | {'avg_q':>12s} | {'blanked':>8s} | {'util':>6s}") 
print(" " + "-" * 60) 

V_values = [0.01, 0.1, 1.0,  5.0, 10.0, 50.0, 100.0] 

for v in V_values: 
    cfg = EnvConfig(
        num_bs=3, num_ue=10, num_prb=25, W=200, K=30, 
        V=v, V_norm=1e9, 
        delta_P_bs=30.0, P0_bs=50.0, 
        arrival_rate_bits=5e4, r_min_bits=1e3, P_avg=200.0, 
        seed=42, enable_jsonl_logger=False,
    )
    env = TwoTimeScaleEnv(cfg) 
    env.reset() 

    act = {"bs_on": np.ones(cfg.num_bs), "bias_db": np.zeros(cfg.num_bs)} 
    last_e, last_q, last_b, last_u = [], [], [], [] 

    for k in range(cfg.K): 
        res = env.step_slow(act) 
        if k >= cfg.K - 10: 
            last_e.append(res.info["energy_window"]) 
            last_q.append(res.info["avg_q_mean"]) 
            last_b.append(res.info["prbs_blanked_frac"]) 
    
    avg_e = np.mean(last_e) 
    avg_q = np.mean(last_q) 
    avg_b = np.mean(last_b) 
    print(f"    {v:>8.2f} | {avg_e:>10.0f} | {avg_q:>12.0f} | {avg_b:>8.3f} | {1-avg_b:6.3f}")


print(f"\n  Expected: energy ⬇ and queues ⬆ as V grows (O(1/V) / O(V))")







