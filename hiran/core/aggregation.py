# hiran/core/aggregation.py 

from __future__ import annotations
import numpy as np 
from .types import KPIs


class WindowAggregator: 
    """Accumulates per-slot metrics over a slow window of W slots. 
    
    Extended for constrained DPP: also tracks virtual queues Z_u, Y, 
    per-BS power, and blank PRB counts. 
    """

    def __init__(self, num_ue: int, num_bs: int, W: int): 
        self.U = num_ue 
        self.B = num_bs 
        self.W = W 
        self.reset() 

    def reset(self, ) -> None: 
        self.sum_thr = np.zeros(self.U, dtype=np.float64) 
        self.sum_q = np.zeros(self.U, dtype=np.float64) 
        self.sum_util = np.zeros(self.B, dtype=np.float64) 
        self.sum_energy = 0.0 
        # --- Constrained DPP additions --- 
        self.sum_Z = np.zeros(self.U, dtype=np.float64) 
        self.sum_Y = 0.0 
        self.sum_power_per_bs = np.zeros(self.B, dtype=np.float64) 
        self.total_prbs_blanked = 0 
        self.total_prbs_possible = 0 
        self._slots = 0 

    def add_slot(self, 
                 thr_u: np.ndarray, 
                 q_u: np.ndarray, 
                 util_b: np.ndarray, 
                 energy_slot: float, 
                 Z_u: np.ndarray, 
                 Y_val: float, 
                 power_per_bs: np.ndarray, 
                 prbs_blanked: int, 
                 num_prb: int) -> None:
        self.sum_thr += thr_u 
        self.sum_q += q_u 
        self.sum_util += util_b 
        self.sum_energy += energy_slot
        # --- Constrained DPP --- 
        self.sum_Z += Z_u 
        self.sum_Y += float(Y_val) 
        self.sum_power_per_bs += power_per_bs 
        self.total_prbs_blanked += prbs_blanked 
        self.total_prbs_possible += num_prb * len(util_b) 
        self._slots += 1 

    def finalize(self, switch_count: int, extra: dict) -> KPIs:
        denom = max(self._slots, 1) 
        prbs_possible = max(self.total_prbs_possible, 1) 
        return KPIs(
            avg_thr_ue=self.sum_thr / denom, 
            avg_q_ue=self.sum_q / denom, 
            util_bs=self.sum_util / denom, 
            energy_window=self.sum_energy, 
            switch_count=int(switch_count), 
            # --- Constrained DPP --- 
            avg_Z_ue=self.sum_Z / denom, 
            avg_Y=self.sum_Y / denom, 
            avg_power_per_bs=self.sum_power_per_bs / denom, 
            prbs_blanked_frac=self.total_prbs_blanked / prbs_possible, 
            extra=dict(extra) 
        )