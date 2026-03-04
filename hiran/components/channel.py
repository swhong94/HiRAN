from __future__ import annotations
import numpy as np 

class SimpleChannel: 
    """
    Large-scale pathloss + optional shadowing 
    Produces per-link gain g_{u, b} (linear) and rsrp_db for association
    """
    def __init__(self, 
                 pathloss_exp: float, 
                 shadowing_std_db: float, 
                 enable_shadowing: bool, 
                 rng: np.random.Generator): 
        self.alpha = float(pathloss_exp) 
        self.sigma_db = float(shadowing_std_db) 
        self.enable_shadowing = bool(enable_shadowing) 
        self.rng = rng 

    def link_gain(self, dist_m: np.ndarray) -> np.ndarray: 
        # dist_m: (U, B) 
        # Simple g ~ d^{-alpha} * 10^{X/10} 
        g = dist_m ** (-self.alpha) 
        if self.enable_shadowing:
            sh_db = self.rng.normal(0.0, self.sigma_db, size=dist_m.shape) 
            g = g * (10.0 ** (sh_db / 10)) 
        return g

    @staticmethod
    def rsrp_db_from_gain(g: np.ndarray, 
                          p_ref_w: float=1.0) -> np.ndarray: 
        # RSRP proxy: 10log10(p_ref * g) 
        return 10.0 * np.log10(np.maximum(p_ref_w * g, 1e-30)) 
    