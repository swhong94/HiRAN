# hiran/components/scheduling.py 

from __future__ import annotations
import numpy as np 

class ConstrainedDPPScheduler: 
    """Constrained DPP scheduler (xApp fast layer). 
    
    Per-slot objective (derived from Lyapunov dript-plus-penalty)
    
        max sum_u (Q_u(t) + Z_u(t)) * r_u(t)  - (V + Y(t) ) * P(t) 
    
    Per-PRB decision under OFDMA (A1) + negligible inter-cell interference (A2): 
    
        For each PRB n on BS b: 
            1. Find u* = argmax_{u \in U_b} (Q_u + Z_u) · P(t) 
            2. Schedule u* on PRB n IFF: 
                (Q_{u*} + Z_{u*}) · r_{u*, b, n} > (V + Y) · ΔP^b / N 
                Otherwise leave PRB n blank (idle) 
    
    This threshold rule is the EXACT solution to the per-slot DPP under A1 + A2 (Neely 2010, Ch. 4) 
    
    Notation: 
        w_u ≜ Q_u + Z_u             (combined weight) 
        V_eff ≜ V + Y               (adaptive penalty) 
    These shorthands appear in code only; paper uses the expanded form. 
    """    
    def __init__(self, V: float=1.0, V_norm: float=1.0): 
        self.V = float(V) 
        self.V_norm = float(V_norm) 

    def step_slot(
        self, 
        Q: np.ndarray,          # (U, ) real queue backlog 
        Z: np.ndarray,          # (U, ) virtual queue for min-rate 
        Y: float,               # scalar virtual queue for power budget 
        assoc: np.ndarray,      # (U, ) user → BS mapping 
        bs_on: np.ndarray,      # (B, ) ON/OFF state 
        rates_ub: np.ndarray,   # (U, B) achievable rate per PRB 
        num_prb: int,           # N – number of PRBs per BS 
        delta_P_per_bs: np.ndarray  # (B, ) dynamic power slope ΔP^b
    ) -> tuple[np.ndarray, np.ndarray, int]: 
        """Run one slot of constrained DPP scheduling. 
        
        Under A1 + A2, per-PRB rates are identical across PRBs for a given 
        (u, b) pair. However, we allocate PRBs **sequentially**: after 
        assigning a PRB to user u*, we debit the served bits from the 
        *working copy of Q_u* so the next PRB may go to a different UE
        whose residual queue weight is now higher. This is standard
        practice for MaxWeight with divisible resources and ensures that 
        service is spread across UEs rather than piling onto one.
        
        The blank-PRB threshold is evaluated per PRB: 
            schedule u* on PRB n iff 
            (tilde{Q}_{u*} + Z_{u*}) · r_{u*, b, n} > (V + Y) · ΔP^b / N 
            
        Where tilde{Q} is the running residual queue after prior PRB 
        allocations within the same slot.
        
        Returns: 
            service_bits:   (U, )   total bits served to each UE this slot 
            util_bs:        (B, )   fraction of PRBs actually scheduled per BS 
            prbs_blanked:   int     total number of PRBs left blank across all BSs
        """
        U, B = rates_ub.shape 
        service = np.zeros(U, dtype=np.float64) 
        util_counts = np.zeros(B, dtype=np.float64)     # PRBs used per BS 
        total_blanked = 0 

        # Working copy of real queue - debited within slot as PRBs are assigned 
        # Virtual queue Z is NOT debited within-slot (it updates once per slot). 
        Q_residual = Q.copy() 

        # Adaptive penalty: V_eff = V·V_norm + Y 
        # V_norm absorbs the typical scale of Q·r·N/ΔP so that V ∈ [0.01, 100] 
        V_eff = self.V * self.V_norm + Y 

        for b in range(B): 
            if bs_on[b] < 0.5: 
                total_blanked += num_prb 
                continue 

            u_idx = np.where(assoc == b)[0] 
            if u_idx.size == 0: 
                total_blanked += num_prb 
                continue 

            # Per-PRB power threshold (same for all PRBs of this BS) 
            threshold = V_eff * delta_P_per_bs[b] / max(num_prb, 1) 

            # Rate each associated UE can get per PRB on this BS 
            r_b = rates_ub[u_idx, b]                # (|U_b|, ) 

            prbs_used = 0 
            for _n in range(num_prb): 
                # Combined weight with residual queue 
                w = (Q_residual[u_idx] + Z[u_idx]) * r_b 

                best_local = int(np.argmax(w)) 
                best_w = w[best_local] 

                if best_w <= threshold: 
                    # All remaining PRBs for this BS will also fail 
                    # (weights can only decrease as we debit more) 
                    total_blanked += (num_prb - _n) 
                    break 

                best_u = u_idx[best_local]
                served = r_b[best_local] 
                service[best_u] += served 
                # Debit from residual Queue (floor at 0 ) 
                Q_residual[best_u] = max(Q_residual[best_u] - served, 0.0) 
                prbs_used += 1 

            util_counts[b] = prbs_used 
        
        # Convert to fraction 
        util_bs = util_counts / max(num_prb, 1) 

        return service, util_bs, total_blanked
    

# --- Legacy alias for backward compatibility --- 
class DPPSchedulerGreedy(ConstrainedDPPScheduler): 
    """Deprecated alias, Use ConstrainedDPPScheduler directly. """
    pass 