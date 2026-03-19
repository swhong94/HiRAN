# hiran/core/logger.py 

from __future__ import annotations
import json
import os 
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np 
from .types import KPIs


class WindowLogger: 
    """Appends one JSON line per slow window to a jsonl file. 
    
    Fields logged: 
        k                           window index 
        reward                      slow-layer reward R_k 
        pf_througput                ∑_u log(hat{R}_u + ε)
        energy_window               ∑_t P(t) over the window 
        switch_count                ∑_b 1[s_{b, k} ≠ s_{b, k-1} 
        avg_Q_per_ue                [list] window-avg real queue per UE 
        avg_Z_per_ue                [list] window-avg virtual queue Z per UE 
        Y_value                     window-avg virtual queue Y 
        avg_thr_per_ue              [list] window-avg throughput per UE 
        util_per_bs                 [list] window-avg utilization per BS 
        avg_power_per_bs            [list] window-avg power per BS 
        prbs_blanked_frac           fraction of PRB-slots left blank 
        capacity_violation_flag     bool - heuristic: any Q_u diverging 
    """

    def __init__(self, log_dir: str, run_name: str="run"):
        self.log_dir = Path(log_dir) 
        self.log_dir.mkdir(parents=True, exist_ok=True) 
        self.path = self.log_dir / f"{run_name}.jsonl" 
        # Truncate on init (new episode) 
        self._fh = open(self.path, "w") 

    def log_window(
        self, 
        k: int, 
        reward: float, 
        pf: float, 
        kpis: KPIs, 
        extra: Optional[Dict[str, Any]] = None 
    ) -> None: 
        # Heuristic capacity violation flag: 
        # If any user's avg queue exceeds a threshold, the topology 
        # is likely pushing λ outside Λ(x_k). 
        q_max = float(np.max(kpis.avg_q_ue)) if kpis.avg_q_ue.size > 0 else 0.0 
        cap_violation = bool(q_max > 1e7)       # conservative threshold 

        record: Dict[str, Any] = { 
            "k": int(k), 
            "reward": float(reward), 
            "pf_throughput": float(pf), 
            "energy_window": float(kpis.energy_window), 
            "switch_count": int(kpis.switch_count), 
            "avg_Q_per_ue": kpis.avg_q_ue.tolist(), 
            "avg_Z_per_ue": kpis.avg_Z_ue.tolist(), 
            "Y_value": float(kpis.avg_Y), 
            "avg_thr_per_ue": kpis.avg_thr_ue.tolist(), 
            "util_per_bs": kpis.util_bs.tolist(), 
            "avg_power_per_bs": kpis.avg_power_per_bs.tolist(), 
            "prbs_blanked_frac": float(kpis.prbs_blanked_frac), 
            "capacity_violation_flag": cap_violation,
        }
        if extra: 
            record.update(extra) 
        self._fh.write(json.dumps(record) + "\n") 
        self._fh.flush() 

    def close(self) -> None: 
        if self._fh and not self._fh.closed: 
            self._fh.close() 
    
    def __del__(self): 
        self.close() 