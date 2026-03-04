from __future__ import annotations
import numpy as np 

class DPPSchedulerGreedy: 
    """
    Greedy Per-BS per-PRB: 
        choose u maximizing Q_u * r_{u, b, n} 
    """
    def __init__(self, V: float=0.0):
        self.V = float(V) 

    def step_slot(self, 
                  Q: np.ndarray, 
                  assoc: np.ndarray, 
                  bs_on: np.ndarray, 
                  rates_ub: np.ndarray, 
                  num_prb: int, ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns: 
            service_bits: (U, ) 
            util_bs: (B, ) fraction of PRBs used (v1 always 1.0 if bs has any UE and is on) 
        """
        U, B = rates_ub.shape 
        service = np.zeros((U, ), dtype=np.float64) 
        util = np.zeros((B, ), dtype=np.float64) 

        # For v1, per-PRB rate is identical, so per-BS decision is same each PRB 
        for b in range(B): 
            if bs_on[b] < 0.5: 
                continue 
            u_idx = np.where(assoc == b)[0] 
            if u_idx.size == 0: 
                continue

            # weight for each UE in this BS 
            w = Q[u_idx] * rates_ub[u_idx, b] 
            # If all zero queues, still pick best rate (tie-breaker) 
            pick = u_idx[int(np.argmax(w))] 

            # allocate all PRBs to that UE (v1) 
            service[pick] += num_prb * rates_ub[pick, b] 
            util[b] = 1.0 
        
        return service, util 
    
