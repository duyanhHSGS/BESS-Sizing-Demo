"""bess_env.py  CMDP environment for forecast-free BESS dispatch.

Design follows the two-layer framework in CoSoLyThuyet_DRL_BESS_Sizing.html
(Hu et al. 2026): the agent sees only real-time measurements (load, PV, SOC,
tariff, running monthly peak)  NO forecast  and outputs one continuous
action a in [-1, 1] mapped to a desired BESS power.

HARD-CONSTRAINT SAFETY PROJECTION (never learned, always enforced):
  * zero export : discharge is capped at the net load, so
                  grid[t] = eff_load + cg - d >= 0 for ANY policy output.
  * SOC bounds  : charge/discharge are capped by the energy head-room in
                  [SOC_min, SOC_max] at the current step.
  * P_rated     : |p_bess| <= P_rated on the AC side.
The RL problem is therefore unconstrained for the learner; the projection
makes the 4 critical scenarios in CLAUDE.md structurally satisfiable.

SPARSE DEMAND-CHARGE SHAPING:
  The monthly demand charge T_cap * max_t D_t (D = 30-min rolling average of
  grid import) is path-dependent and fires once a month. We shape it into a
  dense signal by charging the MARGINAL increment of the running monthly
  peak at each step:  pen_t = T_cap * max(0, D_t - D_run).
  Summed over the month this telescopes to exactly T_cap * (D_peak - D_run0).
  D_run is initialised at d_run_init_frac * p_ref (not 0): any realistic
  monthly peak exceeds that floor, so the telescoped sum differs from the
  true bill only by a policy-independent constant  but the first-day ramp
  no longer produces huge spurious penalty spikes that destabilise the
  value function.

POTENTIAL-BASED SOC SHAPING (Ng et al. 1999  preserves the optimal policy):
  Phi(s) = usable stored energy * eta_dis * price_mid. The agent receives
  gamma*Phi(s') - Phi(s) each step, which gives IMMEDIATE credit for
  storing energy whose payoff (peak discharge) is otherwise ~68 steps away.

COUNTERFACTUAL VARIANCE REDUCTION:
  The raw bill is dominated by the policy-independent no-BESS cost of the
  random load/PV realisation, which buries the controllable signal
  (~5% of the bill) under return variance. The reward therefore pays only
  the DELTA against a no-BESS counterfactual running in lock-step:
    energy term = -tariff * (cg - d)          (charging cost vs. displaced buy)
    peak term   = -(pen_actual - pen_nobess)  (both telescope; the shared
                                               uncontrollable ramp cancels)
  Subtracting policy-independent terms changes no gradients in expectation
  but shrinks advantage variance by an order of magnitude.

PEAK-SAFE CHARGING PROJECTION:
  Grid-charging is capped so it can never itself raise the running monthly
  peak (cg <= max(0, d_run - eff_load)). Charging is deferrable, and at
  T_cap = 235k VND/kW a self-inflicted peak always dominates the ~7k
  VND/kWh/day arbitrage value it could enable, so this cut never removes
  the optimum  and it deletes the catastrophic exploration spikes that
  destabilised early training.
"""
from __future__ import annotations

import numpy as np

from common import DT_HOURS, STEPS_PER_DAY, dt_from_steps_per_day, steps_per_day_from_dt, tariff_vector
from scenario_gen import MonthData

OBS_DIM = 13
OBS_DIM_FC = 17             # forecast-informed variant (+4 features)
REWARD_SCALE = 1e6          # rewards in millions of VND


def _ar1_noise_series(actual: np.ndarray, sigma: float, rho: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Multiplicative AR(1) forecast error  same model Tool A/benchmark
    uses for the SADRBC forecast, so both controllers see equal quality."""
    if sigma <= 0:
        return actual.copy()
    n = len(actual)
    e = np.zeros(n)
    w = sigma * np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + w * rng.standard_normal()
    return np.maximum(0.0, actual * (1.0 + e))


class BESSEnv:
    """Month-long episode using the selected data resolution."""

    def __init__(self, cfg, p_ref_kw: float = 500.0,
                 degradation_vnd_per_kwh: float = 50.0,
                 d_run_init_frac: float = 0.6,
                 d_run_init_kw: float | None = None,
                 gamma: float = 0.995,
                 use_forecast: bool = False,
                 fc_sigma_load: float = 0.05, fc_sigma_pv: float = 0.15,
                 fc_rho: float = 0.9, fc_seed: int = 12345,
                 n_steps: int = STEPS_PER_DAY,
                 dt_hours: float = DT_HOURS):
        # Resolution is derived from configured dt or the supplied day length.
        # (ch  bi bo GREPO). Ngy d liu phi c ng n_steps mu.
        if dt_hours == DT_HOURS and getattr(cfg, "dt", DT_HOURS) != DT_HOURS:
            dt_hours = float(cfg.dt)
            n_steps = steps_per_day_from_dt(dt_hours)
        self.n_steps = int(n_steps)
        self.dt = float(dt_hours)
        cfg.dt = self.dt
        assert abs(self.n_steps * self.dt - 24.0) < 1e-9,             "n_steps  dt phi = 24h"
        # ca s billing 30 pht = bao nhiu mu   phn gii ny
        self.roll_k = max(2, int(round(0.5 / self.dt)))
        self.cfg = cfg
        self.p_ref = float(p_ref_kw)
        self.deg = float(degradation_vnd_per_kwh)
        # Floor khi to nh thng: PHI THP HN nh ti u ca site,
        # nu khng agent khng bao gi nhn tn hiu hc ct nh (bug
        # thc t site Tande: 0.6p_ref = 900 kW > nh ti u ~490 kW).
        # u tin gi tr tuyt i data-driven t trainer; frac l fallback.
        self.d_run_init = (float(d_run_init_kw) if d_run_init_kw is not None
                           else float(d_run_init_frac) * self.p_ref)
        self.gamma = float(gamma)
        # forecast-informed variant: obs gains 4 look-ahead features fed by
        # a NOISY short-horizon forecast (never the actuals)
        self.use_forecast = bool(use_forecast)
        self.obs_dim = OBS_DIM_FC if self.use_forecast else OBS_DIM
        self.fc_sigma_load, self.fc_sigma_pv = fc_sigma_load, fc_sigma_pv
        self.fc_rho = fc_rho
        self._fc_rng = np.random.default_rng(fc_seed)
        self._fc_load = self._fc_pv = None
        base_tar = tariff_vector(cfg)
        self._tar_base = base_tar
        # Ch nht khng cao im (quy tc EVN, bt qua TOU_RULES):
        # vector ring, hon i ti bin ngy trong reset()/step()
        from common import TOU_RULES, cfg_no_peak
        if TOU_RULES.get("sunday_no_peak"):
            sun = tariff_vector(cfg_no_peak(cfg))
            self._tar_sun = sun
        else:
            self._tar_sun = self._tar_base
        self.tariff = self._tar_base
        self.month: MonthData | None = None
        # trajectory logs (filled during an episode)
        self.log_grid: list[np.ndarray] = []
        self.log_soc: list[np.ndarray] = []
        self.log_pbess: list[np.ndarray] = []

    # ------------------------------------------------------------------
    def reset(self, month: MonthData, soc_init: float | None = None) -> np.ndarray:
        self.month = month
        if month.days:
            detected_steps = len(month.days[0].load)
            if detected_steps <= 0:
                raise ValueError("month contains an empty day")
            if detected_steps != self.n_steps:
                self.n_steps = detected_steps
                self.dt = dt_from_steps_per_day(self.n_steps)
                self.cfg.dt = self.dt
                self.roll_k = max(1, int(round(0.5 / self.dt)))
                self._tar_base = tariff_vector(self.cfg)
                from common import TOU_RULES, cfg_no_peak
                self._tar_sun = tariff_vector(cfg_no_peak(self.cfg)) if TOU_RULES.get("sunday_no_peak") else self._tar_base
        self.day = 0
        self.t = 0
        self.soc = float(soc_init if soc_init is not None else self.cfg.SOC_eod)
        self.d_run = self.d_run_init    # running monthly 30-min peak (kW)
        self.g_prev = 0.0               # previous-step grid import (kW)
        self.d_run_nb = self.d_run_init  # counterfactual no-BESS running peak
        self.g_prev_nb = 0.0
        self.log_grid = [np.zeros(self.n_steps) for _ in month.days]
        self.log_soc = [np.zeros(self.n_steps + 1) for _ in month.days]
        self.log_pbess = [np.zeros(self.n_steps) for _ in month.days]
        self._buf = []          # rolling 30' grid tht
        self._buf_nb = []       # rolling 30' grid no-BESS
        self.log_soc[0][0] = self.soc
        self._set_day_tariff()
        self._make_day_forecast()
        return self._obs()

    def _set_day_tariff(self):
        from common import is_sunday
        day = self.month.days[self.day]
        self.tariff = self._tar_sun if is_sunday(day) else self._tar_base

    def _make_day_forecast(self):
        if not self.use_forecast:
            return
        day = self.month.days[self.day]
        self._fc_load = _ar1_noise_series(day.load, self.fc_sigma_load,
                                          self.fc_rho, self._fc_rng)
        self._fc_pv = _ar1_noise_series(day.pv, self.fc_sigma_pv,
                                        self.fc_rho, self._fc_rng)

    def _fc_features(self, t):
        """4 look-ahead features: mean forecast eff-load & PV over the next
        1 h (t+1..t+4) and the following 2 h (t+5..t+12), /p_ref."""
        fc_eff = np.maximum(0.0, self._fc_load - self._fc_pv)
        out = []
        q = self.n_steps // 24  # s mu / gi
        for lo, hi in ((t + 1, t + 1 + q), (t + 1 + q, t + 1 + 3 * q)):
            lo, hi = min(lo, self.n_steps), min(hi, self.n_steps)
            if lo >= hi:                       # end of day  hold last value
                out += [fc_eff[-1] / self.p_ref, self._fc_pv[-1] / self.p_ref]
            else:
                out += [float(fc_eff[lo:hi].mean()) / self.p_ref,
                        float(self._fc_pv[lo:hi].mean()) / self.p_ref]
        return out

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        day = self.month.days[self.day]
        t = self.t
        load, pv = day.load[t], day.pv[t]
        eff = max(0.0, load - pv)
        sur = max(0.0, pv - load)
        t_next = min(t + self.n_steps // 24, self.n_steps - 1)  # 1h ti
        ang = 2.0 * np.pi * t / self.n_steps
        base = [
            np.sin(ang), np.cos(ang),
            eff / self.p_ref,
            pv / self.p_ref,
            sur / self.p_ref,
            self.soc,
            self.tariff[t] / self.cfg.price_peak,
            self.tariff[t_next] / self.cfg.price_peak,
            self.d_run / self.p_ref,
            self.g_prev / self.p_ref,
            1.0 if day.day_type == "working" else 0.0,
            self.day / max(1, len(self.month.days)),
            t / self.n_steps,
        ]
        if self.use_forecast:
            base += self._fc_features(t)
        return np.array(base, dtype=np.float32)

    # ------------------------------------------------------------------
    def project_action(self, a: float, load: float, pv: float):
        """Safety projection. Returns (d, cg, cp) all >= 0, AC kW.
        d = discharge to load, cg = charge from grid, cp = charge from
        free PV surplus. Guarantees grid = eff + cg - d >= 0 and SOC in
        [SOC_min, SOC_max] after the step."""
        cfg = self.cfg
        eff = max(0.0, load - pv)
        sur = max(0.0, pv - load)
        p_des = float(np.clip(a, -1.0, 1.0)) * cfg.P_rated_nominal
        if p_des >= 0.0:                               # discharge request
            avail = max(0.0, (self.soc - cfg.SOC_min)
                        * cfg.E_cap * cfg.eta_dis / self.dt)
            d = min(p_des, cfg.P_rated_nominal, eff, avail)
            return d, 0.0, 0.0
        room = max(0.0, (cfg.SOC_max - self.soc)
                   * cfg.E_cap / (cfg.eta_ch * self.dt))
        ch = min(-p_des, cfg.P_rated_nominal, room)
        cp = min(ch, sur)                              # free PV first
        # peak-safe: grid-charging may never set a new monthly peak itself
        cg_cap = max(0.0, self.d_run - eff)
        cg = min(ch - cp, cg_cap)
        return 0.0, cg, cp

    # ------------------------------------------------------------------
    def step(self, action: float):
        cfg = self.cfg
        day = self.month.days[self.day]
        t = self.t
        load, pv = float(day.load[t]), float(day.pv[t])
        eff = max(0.0, load - pv)

        d, cg, cp = self.project_action(action, load, pv)
        grid = eff + cg - d                            # >= 0 by construction
        soc_before = self.soc
        self.soc = self.soc + (cg + cp) * self.dt * cfg.eta_ch / cfg.E_cap \
                   - d * self.dt / (cfg.eta_dis * cfg.E_cap)
        # numerical guard only  projection already bounds SOC
        self.soc = min(cfg.SOC_max, max(cfg.SOC_min, self.soc))

        # --- costs (actual, for logging/scoring) ------------------------
        energy_cost = self.tariff[t] * grid * self.dt
        self._buf.append(grid)
        self._buf_nb.append(eff)
        if len(self._buf) > self.roll_k:
            self._buf.pop(0)
            self._buf_nb.pop(0)
        if len(self._buf) < self.roll_k:
            d_t = 0.0                    # cha  ca s 30' u ngy
            d_t_nb = 0.0
        else:
            d_t = float(np.mean(self._buf))
            d_t_nb = float(np.mean(self._buf_nb))
        peak_pen = cfg.T_cap * max(0.0, d_t - self.d_run)
        self.d_run = max(self.d_run, d_t)
        peak_pen_nb = cfg.T_cap * max(0.0, d_t_nb - self.d_run_nb)
        self.d_run_nb = max(self.d_run_nb, d_t_nb)
        deg_cost = self.deg * (d + cg + cp) * self.dt

        # --- reward: counterfactual delta vs no-BESS ---------------------
        energy_delta = self.tariff[t] * (cg - d) * self.dt
        peak_delta = peak_pen - peak_pen_nb
        # potential-based shaping: value of stored usable energy at mid price
        phi_coef = cfg.E_cap * cfg.eta_dis * cfg.price_mid
        phi_before = (soc_before - cfg.SOC_min) * phi_coef
        phi_after = (self.soc - cfg.SOC_min) * phi_coef
        shaping = self.gamma * phi_after - phi_before
        reward = (-(energy_delta + peak_delta + deg_cost) + shaping) / REWARD_SCALE

        # --- logs ------------------------------------------------------
        self.log_grid[self.day][t] = grid
        self.log_pbess[self.day][t] = d - (cg + cp)
        self.log_soc[self.day][t + 1] = self.soc
        self.g_prev = grid
        self.g_prev_nb = eff

        # --- advance ---------------------------------------------------
        self.t += 1
        done = False
        if self.t >= self.n_steps:
            self.t = 0
            self.day += 1
            self.g_prev = 0.0        # billing windows do not straddle days
            self.g_prev_nb = 0.0
            self._buf = []
            self._buf_nb = []
            if self.day >= len(self.month.days):
                done = True
            else:
                self.log_soc[self.day][0] = self.soc
                self._set_day_tariff()
                self._make_day_forecast()
        obs = None if done else self._obs()
        info = {"grid_kw": grid, "energy_cost": energy_cost,
                "peak_pen": peak_pen, "peak_pen_nb": peak_pen_nb,
                "d_run": self.d_run}
        return obs, reward, done, info
