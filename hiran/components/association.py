# hiran/components/association.py 

from __future__ import annotations
import numpy as np 

class AssociationManager: 
    def __init__(self, ): 
        pass 

    @staticmethod 
    def associate(rsrp_db: np.ndarray,
                  bias_db: np.ndarray, 
                  bs_on: np.ndarray) -> np.ndarray:
        """Bias-based user association (fixed within each slow window) 

        a_{u, k} = argmax_{b : s_b=1} (RSRP_{u, b} + β_{b, k}) 

        Args: 
            rsrp_db:    (U, B) 
            bias_db:    (B, ) 
            bs_on:      (B, ) in {0, 1} 
        Returns 
            a_u:    (U, ) integer bs index 
        """ 
        U, B = rsrp_db.shape 
        score = rsrp_db + bias_db[None, :] 
        # Mask OFF BS with -inf 
        mask = (bs_on[None, :] > 0.5) 
        score = np.where(mask, score, -np.inf) 

        # Safety: if all BSs are off, fall back to strongest RSRP 
        if not np.any(bs_on > 0.5): 
            return np.argmax(rsrp_db, axis=1).astype(np.int64) 
        
        return np.argmax(score, axis=1).astype(np.int64) 
    
    @staticmethod
    def serving_sets(a_u: np.ndarray, 
                     num_bs: int, ) -> list[np.ndarray]: 
        sets = [] 
        for b in range(num_bs): 
            sets.append(np.where(a_u == b)[0])
        return sets 
    