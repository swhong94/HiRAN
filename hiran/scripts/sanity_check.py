# !/usr/bin/env python 3 

"""Sanity check for the HiRAN environment after modifications 

TESTS: 
    1. Basic end-to-end: reset + step_slow runs without error 
    2. Block 3: all-BS-OFF guard auto-activates best-RSRP BS 
    3. Block 3: bias smoothness clips large jumps 
    4. Block 2: virtual queues Z_u and Y are tracked (non-zero after slots) 
    5. Block 2: blank PRB thresholds causes util < 1.0 when V is larg 
    6. Block 2: load-dependent power responds to utilization 
    7. Block 1: reward includes queue penalty term 
    8. Block 1: logger writes JSONL file 
"""

import sys 
import os 
import json 
import numpy as np 

# Add parent to path so hiran is importable 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiran.core.types import EnvConfig 
from hiran.core.env import TwoTimeScaleEnv 

def test_basic_e2e(): 
    """Test 1: basic end-to-end."""
    cfg = EnvConfig(num_bs=3, num_ue=10, num_prb=25, W=50, K=5, 
                    V=1.0, seed=42, log_dir="/tmp/hiran_test", 
                    enable_jsonl_logger=False)
    env = TwoTimeScaleEnv(cfg) 
    obs = env.reset() 

    for k in range(cfg.K):
        action = {
            "bs_on": np.ones(cfg.num_bs), 
            "bias_db": np.zeros(cfg.num_bs),
        }   
        result = env.step_slow(action) 

    assert result.done, "Episode should be done after K windows" 
    print(f"    [PASS] Test 1: basic end-to-end") 



def test_all_off_guard(): 
    """Test 2: Block 3 – all-BS-OFF auto-correction"""
    cfg = EnvConfig(num_bs=3, num_ue=10, num_prb=25, W=10, K=2, 
                    V=1.0, seed=42, enable_jsonl_logger=False)
    env = TwoTimeScaleEnv(cfg) 
    env.reset() 

    # Propose all BSs OFF 
    action = {
        "bs_on": np.zeros(cfg.num_bs), 
        "bias_db": np.zeros(cfg.num_bs) 
    }
    result = env.step_slow(action) 

    # After guard, at least one BS must be ON 
    assert np.sum(env.bs_on) >= 1.0, \
        f"All-off guard failed: bs_on = {env.bs_on}"
    print(f"    [PASS] Test 2: all-BS-OFF guard activates one BS")


def test_bias_smoothness(): 
    """Test 3: Block 3 – bias smoothness constrains. """
    cfg = EnvConfig(num_bs=3, num_ue=10, num_prb=25, W=10, K=3, 
                    delta_bias_max=2.0, V=1.0, seed=42, 
                    enable_jsonl_logger=False) 
    env = TwoTimeScaleEnv(cfg) 
    env.reset() 

    # bias_db starts at 0.0 for all BSs 

    # Try to jump bias by 6.0 dB in one step (should be clipped to ±2.0) 
    action = {
        "bs_on": np.ones(cfg.num_bs), 
        "bias_db": np.full(cfg.num_bs, 6.0), 
    }
    env.step_slow(action) 
    
    max_bias = float(np.max(np.abs(env.bias_db))) 
    assert max_bias <= 2.0 + 1e-9, \
        f"Bias smoothness failed: max [bias] = {max_bias}, expected=2.0"
    print(f"    [PASS] Test 3: bias smoothness (max |Δβ| - {max_bias:.2f} ≤ 2.0)")


def test_virtual_queues_tracked(): 
    """Test 4: Block 2 - virtual queues Z-u, Y are tracked"""
    cfg = EnvConfig(num_bs=3, num_ue=10, num_prb=25, W=100, K=2, 
                    V=1.0, r_min_bits=1e4, P_avg=120.0, seed=42, 
                    enable_jsonl_logger=False)
    env = TwoTimeScaleEnv(cfg) 
    env.reset() 

    action = {"bs_on": np.ones(cfg.num_bs), "bias_db": np.zeros(cfg.num_bs) }
    result = env.step_slow(action) 

    avg_Z = result.info.get("avg_Z_mean", None) 
    avg_Y = result.info.get("avg_Y", None) 
    assert avg_Z is not None, "avg_Z_mean missing from info" 
    assert avg_Y is not None, "avg_Y missing from info" 
    # Z should be > 0 bvecause r_min_bits > 0 drives it up 
    assert avg_Z > 0, f"Z_u should be > 0 with r_min > 0, got {avg_Z}"
    print(f"    [PASS] Test 4: virtual queues tracked (avg Z = {avg_Z:.1f}, avg Y = {avg_Y:.1f})")


def test_blank_prb_effect(): 
    """Test 5: Block 2 - Large V causes PRB blanking."""
    cfg_low_V = EnvConfig(num_bs=2, num_ue=5, num_prb=25, W=100, K=2, 
                          V=0.001, V_norm=1e9, delta_P_bs=30.0, seed=42, 
                          arrival_rate_bits=5e4, r_min_bits=1e3, 
                          enable_jsonl_logger=False)
    cfg_high_V = EnvConfig(num_bs=2, num_ue=5, num_prb=25, W=100, K=2, 
                           V=1000, V_norm=1e9, delta_P_bs=30.0, seed=42, 
                           arrival_rate_bits=5e4, r_min_bits=1e3, 
                           enable_jsonl_logger=False)
    env_low = TwoTimeScaleEnv(cfg_low_V) 
    env_high = TwoTimeScaleEnv(cfg_high_V) 

    env_low.reset() 
    env_high.reset() 

    act = {"bs_on": np.ones(2), "bias_db": np.zeros(2)} 
    res_low = env_low.step_slow(act) 
    res_high = env_high.step_slow(act) 

    blank_low = res_low.info["prbs_blanked_frac"] 
    blank_high = res_high.info["prbs_blanked_frac"] 

    print(f"    V=0.001 → blanked {blank_low:.2%}, V=1000 → blanked {blank_high:.2%}")
    assert blank_high >= blank_low,  "Higher V should blank at least as many PRBs" 
    print(f"    [PASS] Test 5: higher V causes more PRB blanking") 


def test_load_dependent_power(): 
    """Test 6: Block 2 – power varies with utilization"""
    cfg = EnvConfig(num_bs=2, num_ue=5, num_prb=25, W=50, K=2, 
                    P0_bs=50.0, delta_P_bs=30.0, V=0.001, seed=42, 
                    enable_jsonl_logger=False)

    env = TwoTimeScaleEnv(cfg) 
    env.reset() 

    # All ON, low v → util should be ~1.0 → power = P0 + ΔP = 80W per BS 
    act = {"bs_on": np.ones(2), "bias_db": np.zeros(2)} 
    result = env.step_slow(act) 

    expected_per_bs = cfg.P0_bs + cfg.delta_P_bs    # 80W if util = 1
    actual_energy_per_slot = result.info["energy_window"] / cfg.W 
    # 2 BSs at ~80W each -> ~160W total per slot 
    print(f"    energy/slot = {actual_energy_per_slot:.1f}W "
          f" (expect = {2 * expected_per_bs:0f}W with 2BSs)")
    assert actual_energy_per_slot > 2 * cfg.P0_bs, \
        "Power should exceed 2 x P0 when BSs are fully utilized" 
    print(f"    [PASS] Test 6: load-dependent power model works") 


def test_queue_penalty_in_reward(): 
    """Test 7: Block 1 – reward includes queue penalty"""
    # Run two envs: one with eta_queue =0 , one with eta_queue=1e-3 
    cfg_no_q = EnvConfig(num_bs=2, num_ue=5, num_prb=25, W=50, K=2, 
                         V=1.0, eta_queue=0.0, seed=42, 
                         enable_jsonl_logger=False)
    cfg_with_q = EnvConfig(num_bs=2, num_ue=5, num_prb=25, W=50, K=2,
                           V=1.0, eta_queue=1e-3, seed=42, 
                           enable_jsonl_logger=False)
    
    env_no = TwoTimeScaleEnv(cfg_no_q)
    env_with = TwoTimeScaleEnv(cfg_with_q) 

    env_no.reset() 
    env_with.reset() 

    act = {"bs_on": np.ones(2), "bias_db": np.zeros(2)} 
    res_no = env_no.step_slow(act) 
    res_with = env_with.step_slow(act) 

    # with queue penalty, reward should be lower (penalty subtracts) 
    print(f"    reward (no_queue): {res_no.reward:.2f}") 
    print(f"    reward (with_queue): {res_with.reward:.2f}") 
    assert res_with.reward <= res_no.reward + 1e-9, \
        "Queue penalty should lower the reward" 
    print(f"    [PASS] Test 7: queue penalty term active in reward") 


def test_logger_writes():
    """Test 8: Block 1 - JSONL logger creates file."""
    log_dir = "tmp/rian_test_logger" 
    cfg = EnvConfig(num_bs=2, num_ue=5, num_prb=25, W=20, K=3, 
                    V=1.0, seed=42, log_dir=log_dir, 
                    enable_jsonl_logger=True) 
    env = TwoTimeScaleEnv(cfg) 
    env.reset() 

    act = {"bs_on": np.ones(2), "bias_db": np.zeros(2)} 

    for _ in range(cfg.K): 
        env.step_slow(act) 
    
    # Check file exists and has K lines 
    log_path = os.path.join(log_dir, f"seed{cfg.seed}.jsonl") 
    assert os.path.exists(log_path), f"Log file not found: {log_path}" 

    with open(log_path) as f: 
        lines = f.readlines() 
    
    assert len(lines) == cfg.K, f"Expected {cfg.K} log lines, got {len(lines)}" 

    # Parse one line to check fields 
    record = json.loads(lines[0]) 
    required_fields = [
        "k", "reward", "pf_throughput", "energy_window", "switch_count", 
        "avg_Q_per_ue", "avg_Z_per_ue", "Y_value", "prbs_blanked_frac", 
        "capacity_violation_flag"
    ]
    for field in required_fields:
        assert field in record, f"Missing field '{field}' in log" 

    print(f"    [PASS] Test 8: JSONL logger wrote {len(lines)} lines with all fields") 


def test_v_tradeoff(): 
    """Quick V-sweep sanity: energy should decrease, queues increase with V."""
    V_values = [0.01, 5.0, 100.0] 
    energies = [] 
    avg_queues = [] 

    for V in V_values: 
        cfg = EnvConfig(num_bs=3, num_ue=10, num_prb=25, W=200, K=15, 
                        V=V, V_norm=1e9, delta_P_bs=30.0, P0_bs=50.0, 
                        arrival_rate_bits=5e4, r_min_bits=1e3, P_avg=200.0,
                        seed=42, enable_jsonl_logger=False)
        env = TwoTimeScaleEnv(cfg) 
        env.reset() 

        total_energy = 0.0 
        total_q = 0.0

        act = {"bs_on": np.ones(cfg.num_bs), "bias_db": np.zeros(cfg.num_bs)}
        for _ in range(cfg.K): 
            res = env.step_slow(act) 
            total_energy += res.info["energy_window"] 
            total_q += res.info["avg_q_mean"] 
        
        energies.append(total_energy / cfg.K) 
        avg_queues.append(total_q / cfg.K) 

    
    print(f"    V = {V_values[0]:.2f} → energy={energies[0]:.0f}, queue={avg_queues[0]:0f}")
    print(f"    V = {V_values[1]:.2f} → energy={energies[1]:.0f}, queue={avg_queues[1]:0f}")
    print(f"    V = {V_values[2]:.2f} → energy={energies[2]:.0f}, queue={avg_queues[2]:0f}")

    # Energy should decrease or stay as V grows (more blanking) 
    assert energies[-1] <= energies[0] + 1e-3, \
        f"Energy should decrease with V: {energies}" 
    
    # Queues should increase as V grows (DPP tradeoff) 
    assert avg_queues[-1] >= avg_queues[0] - 1e-3, \
        f"Queues should increase with V: {avg_queues}" 
    print(f"    [PASS] V-tradeoff: energy ⬇, queues ⬆ as V grows") 





if __name__ == "__main__": 
    print("=" * 60) 
    print("HiRAN Environment - Sanity checks (Blocks 1-2-3)") 
    print("=" * 60) 

    tests = [
        test_basic_e2e, 
        test_all_off_guard, 
        test_bias_smoothness, 
        test_virtual_queues_tracked, 
        test_blank_prb_effect, 
        test_load_dependent_power, 
        test_queue_penalty_in_reward, 
        test_logger_writes, 
        test_v_tradeoff
    ]

    passed = 0
    failed = 0 
    for t in tests: 
        try: 
            print(f"\n{t.__doc__.strip()}") 
            t() 
            passed += 1 
        except Exception as e: 
            print(f"    [FAIL] {e}") 
            failed += 1 
    
    print(f"\n{'=' * 60}") 
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}") 
    print("=" * 60) 





