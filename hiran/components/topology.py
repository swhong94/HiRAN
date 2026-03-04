# hiran/components/topology.py 

from __future__ import annotations 
from dataclasses import dataclass
import numpy as np 


@dataclass
class Topology: 
    bs_xy: np.ndarray       # (B, 2) 
    ue_xy: np.ndarray       # (U, 2) 

    @staticmethod
    def random(num_bs: int, num_ue: int, area_size_m: float, rng: np.random.Generator) -> "Topology": 
        bs_xy = rng.uniform(0.0, area_size_m, size=(num_bs, 2))
        ue_xy = rng.uniform(0.0, area_size_m, size=(num_ue, 2))
        return Topology(bs_xy=bs_xy, ue_xy=ue_xy)
    
    def distances(self) -> np.ndarray: 
        # (U, B) 
        diff = self.ue_xy[:, None, :] - self.bs_xy[None, :, :] 
        d = np.sqrt((diff ** 2).sum(axis=-1)) 
        return np.maximum(d, 1.0)   # avoid zero distance