# hiran/core/env.py 

from __future__ import annotations
from typing import Any, Tuple, Dict, Optional
import numpy as np 

from .types import EnvConfig, StepResult, KPIs, clamp
from .timeline import Timeline
from .aggregation import WindowAggregator
from .logger import WindowLogger

from ..components.topology import Topology
from ..components.traffic import PoissonTraffic
from ..components.queues import BitQueues, VirtualQueues
from ..components.channel import SimpleChannel
from ..components.association import AssociationManager
from ..components.scheduling import ConstrainedDPPScheduler
from ..components.energy import LoadDependentEnergyModel




class TwoTimeScaleEnv:
    """Two-time-scale HiRAN environment. 

    Slow layer (rApp, window timescale): 
        - BS sleeping control   s_{b, k} ∈ {0, 1} 
        - CIO bias control      β_{b, k} ∈ [β_min, β_max] 
        - Bias-based user association (fixed within window) 

    Fast layer (xApp, slot timescale): 
        - Contrained DPP scheduler with: 
            * Real queue Q_u(t)  -> queue stability     (F-C1) 
            * Virtual queue Z_u(t) -> minimum rate guarantee (F-C2) 
            * Virtual queue Y(t) -> power budget constraint (F-C3) 
        - Per-PRB blank threshold under A1 + A2 
        - Load-dependent power: P_b(t) = s_b · (P_0^b + ΔP^v · util_b(t))

    Slow reward: 
        R_k = ∑_u log(hat{R}_u + ε) - η · E_k - η_sw · S_k - η_Q · ∑_u bar{Q}_u(k)      
    """

    def __init__(self, cfg: EnvConfig): 
        self.cfg = cfg 
        self.rng = np.random.default_rng(cfg.seed) 

        # --- Timeline --- 
        self.time = Timeline(W=cfg.W, K=cfg.K) 

        # --- Components --- 
        self.topo = Topology.random(num_bs=cfg.num_bs, 
                                    num_ue=cfg.num_ue, 
                                    area_size_m=cfg.area_size_m, 
                                    rng=self.rng)
        self.traffic = PoissonTraffic(arrival_rate_bits=cfg.arrival_rate_bits, 
                                      rng=self.rng)
        self.queues = BitQueues(num_ue=cfg.num_ue, 
                                init_bits=cfg.queue_init_bits)
        self.virtual_queues = VirtualQueues(num_ue=cfg.num_ue)
        self.channel = SimpleChannel(pathloss_exp=cfg.pathloss_exp, 
                                     shadowing_std_db=cfg.shadowing_std_db, 
                                     enable_shadowing=cfg.enable_shadowing, 
                                     rng=self.rng)
        self.assoc_mng = AssociationManager() 
        self.scheduler = ConstrainedDPPScheduler(V=cfg.V, V_norm=cfg.V_norm)
        
        # --- Energy model: Björnson two-componenet --- 
        B = cfg.num_bs
        P0_arr = np.full(B, cfg.P0_bs, dtype=np.float64)
        delta_P_arr = np.full(B, cfg.delta_P_bs, dtype=np.float64)
        self.delta_P_per_bs = delta_P_arr 
        self.energy = LoadDependentEnergyModel(P0_per_bs=P0_arr, 
                                               delta_P_per_bs=delta_P_arr, 
                                               p_sleep_w=cfg.p_sleep_w)
        
        # --- Derived static channel ---
        self.dist_ub = self.topo.distances()            # (U, B) 
        self.g_ub = self.channel.link_gain(self.dist_ub) 
        self.rsrp_db = self.channel.rsrp_db_from_gain(g=self.g_ub)

        # --- Window Aggregator --- 
        self.agg = WindowAggregator(num_ue=cfg.num_ue, 
                                    num_bs=cfg.num_bs, 
                                    W=cfg.W)
        
        # --- Slow state --- 
        self.bs_on = np.ones((B, ), dtype=np.float64) 
        self.bias_db = np.zeros((B, ), dtype=np.float64) 
        self.assoc = np.zeros((cfg.num_ue, ), dtype=np.int64)
        self.prev_bs_on = self.bs_on.copy() 

        # --- Logger --- 
        self._logger: Optional[WindowLogger] = None 
        if cfg.enable_jsonl_logger:
            self._logger = WindowLogger(log_dir=cfg.log_dir, 
                                        run_name=f"seed{cfg.seed}")
            

    # ===================================================================
    #   RESET 
    # ===================================================================
    def reset(self, seed: Optional[int]=None) -> Dict[str, np.ndarray]:
        if seed is not None: 
            self.rng = np.random.default_rng(seed=seed)
        self.time.reset() 

        # Reset all queues
        self.queues.reset(self.cfg.queue_init_bits)
        self.virtual_queues.reset() 

        # Reset Slow controls 
        B = self.cfg.num_bs
        self.bs_on = np.ones((B, ), dtype=np.float64) 
        self.bias_db = np.zeros((B, ), dtype=np.float64)
        self.prev_bs_on = self.bs_on.copy() 

        # Resample shadowing each episode 
        self.g_ub = self.channel.link_gain(dist_m=self.dist_ub) 
        self.rsrp_db = self.channel.rsrp_db_from_gain(g=self.g_ub)

        # Reset aggregator 
        self.agg.reset() 

        # Associate for window 0 
        self.assoc = self.assoc_mng.associate(rsrp_db=self.rsrp_db, 
                                              bias_db=self.bias_db, 
                                              bs_on=self.bs_on) 
        
        return self._build_obs(kpis=None) 
    
    # ===================================================================
    #   STEP SLOW 
    # ===================================================================
    def step_slow(self, slow_action: Dict[str, np.ndarray]) -> StepResult: 
        """Advance one slow window (W fast slots)

        slow_action: 
            {
                "bs_on": (B, ) float in {0, 1} or [0, 1] 
                "bias_db":  (B, ) float in [bias_min, bias_max]
            }
        """
        cfg = self.cfg 

        # --- Parse and threshold BS ON/OFF -- 
        new_bs_on = slow_action.get("bs_on", self.bs_on).astype(np.float64) 
        new_bs_on = (new_bs_on >= 0.5).astype(np.float64) 

        # --- ALL-BS-OFF guard --- 
        if np.sum(new_bs_on) < 0.5: 
            # Auto-activate the BS with best aggregate RSRP 
            rsrp_sum_per_bs = np.sum(self.rsrp_db, axis=0) 
            b_star = int(np.argmax(rsrp_sum_per_bs)) 
            new_bs_on[b_star] = 1.0 
        
        # --- Parse bias --- 
        new_bias = slow_action.get("bias_db", self.bias_db).astype(np.float64) 

        # --- Bias smoothness constraint --- 
        #   |β_{b, k+1} - β_{b, k}| ≤ Δ_max 
        bias_delta = new_bias - self.bias_db 
        bias_delta = np.clip(bias_delta, -cfg.delta_bias_max, cfg.delta_bias_max) 
        new_bias = self.bias_db + bias_delta  
        
        # Global bias clamp  
        new_bias = clamp(new_bias, lo=cfg.bias_min_db, hi=cfg.bias_max_db) 

        # --- Switching count --- 
        switch_count = int(np.sum((new_bs_on >= 0.5) != (self.bs_on >= 0.5)))

        # --- Update slow state --- 
        self.prev_bs_on = self.bs_on.copy() 
        self.bs_on = new_bs_on 
        self.bias_db = new_bias 

        # --- Recompute association for this time window --- 
        self.assoc = self.assoc_mng.associate(rsrp_db=self.rsrp_db, 
                                              bias_db=self.bias_db, 
                                              bs_on=self.bs_on)
        
        # --- Run fast loop for W slots --- 
        self.agg.reset() 
        for _ in range(self.cfg.W): 
            self._step_slot_internal() 
            self.time.step_slot() 

        # --- Finalize window KPIs --- 
        kpis = self.agg.finalize(switch_count=switch_count, extra={})

        # --- Compute slow reward --- 
        #       R_k = ∑_u log(R_u + ε) - η E_k - η_sw S_k - η_Q ∑_u Q_u(k) 
        thr = kpis.avg_thr_ue 
        pf = float(np.sum(np.log(thr + self.cfg.eps_pf)))
        reward = pf - self.cfg.eta_energy * kpis.energy_window - self.cfg.eta_switch * kpis.switch_count -self.cfg.eta_queue * float(np.sum(kpis.avg_q_ue)) 

        done = self.time.episode_done()
        obs = self._build_obs(kpis=kpis) 

        info: Dict[str, Any] = { 
            "k": self.time.k, 
            "pf": pf, 
            "energy_window": kpis.energy_window, 
            "switch_count": kpis.switch_count, 
            "avg_thr_mean": float(np.mean(kpis.avg_thr_ue)), 
            "avg_q_mean": float(np.mean(kpis.avg_q_ue)),
            "avg_Z_mean": float(np.mean(kpis.avg_Z_ue)), 
            "avg_Y": float(kpis.avg_Y), 
            "prbs_blanked_frac": float(kpis.prbs_blanked_frac), 
        }

        # --- Log window --- 
        if self._logger:
            self._logger.log_window(
                k=self.time.k, reward=reward, pf=pf, kpis=kpis)

        return StepResult(obs=obs, reward=reward, done=done, info=info)
    
    # ======================================================================
    # FAST SLOT INTERNAL 
    # ====================================================================== 
    def _step_slot_internal(self, ) -> None: 
        cfg = self.cfg 

        # --- Arrivals --- 
        A = self.traffic.sample(self.cfg.num_ue)    # (U, ) 

        # --- Achievable rate per PRB --- 
        # r_{u, b} = B_{prb} log2(1 + SNR_{u, b}) 
        # SNR = (P_prb * g_{u, b}) / N0     
        P_prb = (self.cfg.p_bs_max_w / self.cfg.num_prb)
        snr_ub = (P_prb * self.g_ub) / max(self.cfg.noise_power_w, 1e-30) 
        r_per_prb_ub = self.cfg.prb_bw_hz * np.log2(1.0 + snr_ub)

        # --- Constrained DPP scheduling --- 
        service, util_bs, prbs_blanked = self.scheduler.step_slot(
            Q=self.queues.Q, 
            Z=self.virtual_queues.Z, 
            Y=self.virtual_queues.Y, 
            assoc=self.assoc, 
            bs_on=self.bs_on, 
            rates_ub=r_per_prb_ub, 
            num_prb=cfg.num_prb, 
            delta_P_per_bs=self.delta_P_per_bs,
        )

        # --- Update Real Queues --- 
        self.queues.update(service_bits=service, arrival_bits=A) 

        # --- Compute power (Load-dependent) --- 
        p_total, p_per_bs = self.energy.power_slot(bs_on=self.bs_on, 
                                                   util_bs=util_bs)

        # --- Update virtual queues --- 
        self.virtual_queues.update_Z(service_bits=service,
                                     r_min=cfg.r_min_bits)
        self.virtual_queues.update_Y(p_total=p_total, 
                                     p_avg=cfg.P_avg)

        # --- Aggregate Window KPIs ---
        self.agg.add_slot(
            thr_u=service, 
            q_u=self.queues.Q.copy(), 
            util_b=util_bs, 
            energy_slot=p_total, 
            Z_u=self.virtual_queues.Z.copy(), 
            Y_val=self.virtual_queues.Y, 
            power_per_bs=p_per_bs, 
            prbs_blanked=prbs_blanked, 
            num_prb=cfg.num_prb,
        )


    # ====================================================================
    #   OBSERVATION BUILDER 
    # ====================================================================
    def _build_obs(self, kpis: Optional[KPIs]) -> Dict[str, np.ndarray]: 
        B = self.cfg.num_bs 
        U = self.cfg.num_ue 

        if kpis is None: 
            avg_thr = np.zeros((U, ), dtype=np.float64) 
            avg_q = self.queues.Q.copy() 
            util = np.zeros((B, ), dtype=np.float64) 
            energy_w = 0.0 
            switch = 0.0 
            avg_Z = np.zeros(U, dtype=np.float64) 
            avg_Y = 0.0 
        else: 
            avg_thr = kpis.avg_thr_ue
            avg_q = kpis.avg_q_ue 
            util = kpis.util_bs 
            energy_w = kpis.energy_window 
            switch = kpis.switch_count 
            avg_Z = kpis.avg_Z_ue 
            avg_Y = kpis.avg_Y
        
        # Action mask: Disallow all off 
        mask_bs_on = np.ones((B, ), dtype=np.float64) 
        # Can be extended to a structured mask 

        obs = {
            # -- BS-level -- 
            "bs_on_prev": self.prev_bs_on.astype(np.float64), 
            "bs_on": self.bs_on.astype(np.float64), 
            "bias_db": self.bias_db.astype(np.float64),
            "util_bs": util.astype(np.float64),
            # -- UE-level -- 
            "avg_thr_ue": avg_thr.astype(np.float64),
            "avg_q_ue": avg_q.astype(np.float64), 
            "avg_z_ue": avg_Z.astype(np.float64), 
            # -- Global -- 
            "energy_window": np.array([energy_w], dtype=np.float64), 
            "switch_count": np.array([switch], dtype=np.float64),
            "Y_value": np.array([avg_Y], dtype=np.float64), 
            # -- Action mask (placeholder; extend in wrapper) -- 
            "action_mask_bs_on": np.ones(B, dtype=np.float64)  
        }
        return obs 
    





