# hiran/core/timeline.py

from __future__ import annotations
from dataclasses import dataclass

@dataclass 
class Timeline: 
    W: int 
    K: int 

    t: int = 0      # slot 
    k: int = 0      # window 

    def reset(self) -> None: 
        self.t = 0 
        self.k = 0 

    def step_slot(self) -> None: 
        self.t += 1 
        self.k = self.t // self.W 

    def window_done(self) -> bool: 
        # True if we just finished a window boundary (after stepping slot) 
        return (self.t % self.W) == 0 
    
    def episode_done(self) -> bool: 
        return self.k >= self.K 
    

