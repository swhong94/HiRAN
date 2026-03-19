# hiran/scenarios/cellular_sleep_bias.py 

from __future__ import annotations 
from dataclasses import dataclass
from ..core.types import EnvConfig
from ..core.env import TwoTimeScaleEnv

@dataclass
class CellularSleepBiasConfig(EnvConfig): 
    pass 

def make_env(cfg: CellularSleepBiasConfig) -> TwoTimeScaleEnv: 
    return TwoTimeScaleEnv(cfg) 

