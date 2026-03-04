from __future__ import annotations
from typing import Any, Tuple, Dict
import numpy as np 

from .types import EnvConfig, StepResult, clamp
from .timeline import Timeline
from .aggregation import WindowAggregator

from ..components.topology import Topology
from ..components.traffic import PoissonTraffic
from ..components.queues import BitQueues
from ..components.channel import SimpleChannel
from ..components.association import AssociationManager
from ..components.scheduling import DPPSchedulerGreedy
from .. components.energy import SimpleEnergyModel




class TwoTimeScaleEnv:
    """
    Minimal two-time-scale env:
        - step_slow (action) advances exactly one window (W slots) 
        - fast loop uses DPP scheduler (internal) 
    """

    def __init__(self, cfg: EnvConfig): 
        self.cfg = cfg 
        self.rng = np.random.default_rng(cfg.seed) 

        # Timeline 
        self.time = Timeline(W=cfg.W, K=cfg.K) 

        # Components 
        self.topo = Topology.random(num_bs=cfg.num_bs, 
                                    num_ue=cfg.num_ue, 
                                    area_size_m=cfg.area_size_m, 
                                    rng=self.rng)
        self.traffic = PoissonTraffic(arrival_rate_bits=cfg.arrival_rate_bits, 
                                      rng=self.rng)
        self.queues = BitQueues(num_ue=cfg.num_ue, 
                                init_bits=cfg.queue_init_bits)
        self.channel = SimpleChannel(pathloss_exp=cfg.pathloss_exp, 
                                     shadowing_std_db=cfg.shadowing_std_db)
        self.assoc_mng = AssociationManager() 
        self.scheduler = DPPSchedulerGreedy(V=cfg.V) 
        self.energy = SimpleEnergyModel(p_on_w=cfg.p_on_w, 
                                        p_sleep_w=cfg.p_sleep_w, 
                                        load_coeff=cfg.load_power_coeff)
        
        # Derived static-ish 
        self.dist_ub = self.topo.distances()            # (U, B) 
        self.g_ub = self.channel.link_gain(self.dist_ub) 
        self.rsrp_db = self.channel.rsrp_db_from_gain(g=self.g_ub)

        # Window aggregation 
        self.agg = WindowAggregator(num_ue=cfg.num_ue, 
                                    num_bs=cfg.num_bs, 
                                    W=cfg.W)
        
        # Slow state 
        self.bs_on = np.ones((cfg.num_bs, ), dtype=np.float64) 
        self.bias_db = np.ones((cfg.num_bs, ), dtype=np.float64) 
        self.assoc = np.zeros((cfg.num_ue, ), dtype=np.int64)
        self.prev_bs_on = self.bs_on.copy() 

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        if seed is not None: 
            self.rng = np.random.default_rng(seed=seed)
        self.time.reset() 

        # Reset Queues 
        self.queues.reset(self.cfg.queue_init_bits)

        # Reset Slow controls 
        self.bs_on = np.ones((self.cfg.num_bs, ), dtype=np.float64) 
        self.bias_db = np.ones((self.cfg.num_bs, ), dtype=np.float64)
        
        self.prev_bs_on = self.bs_on.copy() 

        # (Optional) Resample shadowing each episode by recomputing g_ub 
        self.g_ub = self.channel.link_gain(dist_m=self.dist_ub) 
        self.rsrp_db = self.channel.rsrp_db_from_gain(g=self.g_ub)

        # Reset aggregator 
        self.agg.reset() 

        # Associate for window 0 
        self.assoc = self.assoc_mng.associate(rsrp_db=self.rsrp_db, 
                                              bias_db=self.bias_db, 
                                              bs_on=self.bs_on) 
        
        return self._build_obs(kpis=None) 
    
    def step_slow(self, slow_action: Dict[str, np.ndarray]) -> StepResult: 
        """
        slow_action: 
            {
                "bs_on": (B, ) float in {0, 1} or [0, 1] 
                "bias_db":  (B, ) float in [bias_db_min, bias_db_max]
            }
        """
        # Apply slow action
        new_bs_on = slow_action.get("bs_on", self.bs_on).astype(np.float64) 
        new_bs_on = (new_bs_on >= 0.5).astype(np.float64) 

        new_bias = slow_action.get("bias_db", self.bias_db).astype(np.float64) 
        new_bias = clamp(new_bias, lo=self.cfg.bias_min_db, hi=self.cfg.bias_max_db)

        # Switching count 
        switch_count = int(np.sum((new_bs_on >= 0.5) != (self.bs_on >= 0.5)))

        # Update values 
        self.prev_bs_on = self.bs_on.copy() 
        self.bs_on = new_bs_on 
        self.bias_db = new_bias 

        # Recompute association for this time window 
        self.assoc = self.assoc_mng.associate(rsrp_db=self.rsrp_db, 
                                              bias_db=self.bias_db, 
                                              bs_on=self.bias_db)
        
        # Run fast loop for window W 
        self.agg.reset() 
        for _ in range(self.cfg.W): 
            self._step_slot_internal() 
            self.time.step_slot() 

        # Finalize KPIs 
        kpis = self.agg.finalize(switch_count=switch_count, extra={})

        # Compute slow reward (PF - energy - switching) 
        thr = kpis.avg_thr_ue 
        pf = float(np.sum(np.log(thr + self.cfg.eps_pf)))
        reward = pf - self.cfg.eta_energy * kpis.energy_window - self.cfg.eta_switch * kpis.switch_count

        done = self.time.episode_done()
        obs = self._build_obs(kpis=kpis) 

        info: Dict[str, Any] = { 
            "k": self.time.k, 
            "pf": pf, 
            "energy_window": kpis.energy_window, 
            "switch_count": kpis.switch_count, 
            "avg_thr_mean": float(np.mean(kpis.avg_thr_ue)), 
            "avg_q_mean": float(np.mean(kpis.avg_q_ue))
        }

        return StepResult(obs=obs, reward=reward, info=info)
    
    # ----------------- INTERNALS -------------------- 
    def _step_slot_internal(self, ) -> None: 
        # Arrivals 
        A = self.traffic.sample(self.cfg.num_ue)    # (U, ) 

        # Achievable rate per PRB: r_{u, b} = B_prb * log2(1 + SNR) 
        # Simple SNR: (P_prb * g) / N0 
        # Use equal power per PRB if BS is on; v1 ignores interferencee 
        P_prb = (self.cfg.p_bs_max_w / self.cfg.num_prb)
        snr_ub = (P_prb * self.g_ub) / max(self.cfg.noise_power_w, 1e-30) 
        r_per_prb_ub = self.cfg.prb_bw_hz * np.log2(1.0 + snr_ub)

        # DPP Scheduling 
        service, util_bs = self.scheduler.step_slot(
            Q=self.queues.Q, 
            assoc=self.assoc,
            bs_on=self.bs_on, 
            rates_ub=r_per_prb_ub, 
            num_prb=self.cfg.num_prb, 
        )

        # Update Queues 
        self.queues.update(service_bits=service, arrival_bits=A) 

        # Energy 
        p_slot = self.energy.power_slot(bs_on=self.bs_on, util_bs=util_bs) 

        # Aggregate Window KPIS
        self.agg.add_slot(
            thr_u=service, 
            q_u=self.queues.Q.copy(), 
            util_b=util_bs, 
            energy_slot=p_slot
        )


    def _build_obs(self, kpis) -> Dict[str, np.ndarray]: 
        # Slow observation as a dict (MARL Friendly)
        B = self.cfg.num_bs 
        U = self.cfg.num_ue 

        if kpis is None: 
            avg_thr = np.zeros((U, ), dtype=np.float64) 
            avg_q = self.queues.Q.copy() 
            util = np.zeros((B, ), dtype=np.float64) 
            energy_w = 0.0 
            switch = 0.0 
        else: 
            avg_thr = kpis.avg_thr 
            avg_q = kpis.avg_q_ue 
            util = kpis.util_bs 
            energy_w = kpis.energy_window 
            switch = kpis.switch_count 
        
        # Action mask: Disallow all off 
        mask_bs_on = np.ones((B, ), dtype=np.float64) 
        # Can be extended to a structured mask 

        obs = {
            "bs_on_prev": self.prev_bs_on.astype(np.float64), 
            "bs_on": self.bs_on.astype(np.float64), 
            "bias_db": self.bias_db.astype(np.float64),
            "util_bs": util.astype(np.float64),
            "avg_thr_ue": avg_thr.astype(np.float64),
            "avg_q_ue": avg_q.astype(np.float64), 
            "energy_window": np.array([energy_w], dtype=np.float64), 
            "switch_count": np.array([switch], dtype=np.floatt64),
            "action_mask_bs_on": mask_bs_on, 
        }
        return obs 
    





