from __future__ import annotations
import numpy as np 
from .types import KPIs


class WindowAggregator: 
    def __init__(self, 
                 num_ue: int, 
                 num_bs: int, 
                 W: int): 
        self.U = num_ue 
        self.B = num_bs 
        self.W = W 
        self.reset() 

    def reset(self) -> None: 
        self.sum_thr = np.zeros((self.U, ), dtype=np.float64) 
        self.sum_q = np.zeros((self.U, ), dtype=np.float64) 
        self.sum_util = np.zeros((self.B, ), dtype=np.float64) 
        self.sum_energy = 0.0 
        self._slots = 0 

    def add_slot(self, 
                 thr_u: np.ndarray, 
                 q_u: np.ndarray, 
                 util_b: np.ndarray, 
                 energy_slot: float) -> None:
        self.sum_thr += thr_u 
        self.sum_q += q_u 
        self.sum_util += util_b 
        self.sum_energy += float(energy_slot) 
        self._slots += 1 
    
    def finalize(self, switch_count: int, extra: dict) -> KPIs: 
        denom = max(self._slots, 1) 
        return KPIs(
            avg_thr_ue=self.sum_thr / denom, 
            avg_q_ue=self.sum_q / denom, 
            util_bs=self.sum_util / denom, 
            energy_window=self.sum_energy, 
            switch_count=int(switch_count), 
            extra=dict(extra)
        )