"""bess_env.py  CMDP environment for reactive or real-forecast BESS dispatch.

Design follows the two-layer framework in CoSoLyThuyet_DRL_BESS_Sizing.html
(Hu et al. 2026). Policies use 13 reactive inputs or 17 inputs with four
causal forecast features. Schema v2 keeps those compact dimensions and
reuses the redundant linear time field for next-cheap-window shortage
pressure; sin/cos already encode the full daily cycle. Policies may output
a continuous action in [-1, 1], which the environment converts to a binary
command: -1 for charging or +1 for discharging.

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
from settings import PPO_GAMMA

OBS_DIM = 13
OBS_DIM_FC = 17             # forecast-informed variant (+4 features)
OBS_SCHEMA_VERSION = 2
REWARD_SCALE = 1e6          # rewards in millions of VND


class BESSEnv:
    """Month-long episode using the selected data resolution."""

    def __init__(self, cfg, p_ref_kw: float = 500.0,
                 degradation_vnd_per_kwh: float = 50.0,
                 d_run_init_frac: float = 0.6,
                 d_run_init_kw: float | None = None,
                 gamma: float = PPO_GAMMA,
                 control_dt_minutes: float | None = None,
                 use_forecast: bool = False,
                 use_tariff_lookahead: bool = False,
                 n_steps: int = STEPS_PER_DAY,
                 dt_hours: float = DT_HOURS,
                 record_trajectory: bool = True,
                 extra_obs_dim: int = 0):
        # Resolution is derived from configured dt or the supplied day length.
        # (ch  bi bo GREPO). Ngy d liu phi c ng n_steps mu.
        if dt_hours == DT_HOURS and getattr(cfg, "dt", DT_HOURS) != DT_HOURS:
            dt_hours = float(cfg.dt)
            n_steps = steps_per_day_from_dt(dt_hours)
        self.n_steps = int(n_steps)
        self.dt = float(dt_hours)
        self.control_dt_minutes = (
            float(control_dt_minutes)
            if control_dt_minutes is not None
            else self.dt * 60.0
        )
        self.native_steps_per_action = 1
        cfg.dt = self.dt
        assert abs(self.n_steps * self.dt - 24.0) < 1e-9,             "n_steps  dt phi = 24h"
        self._configure_control_interval()
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
        self.record_trajectory = bool(record_trajectory)
        # Forecast mode accepts only externally prepared causal predictions.
        # Missing predictions are an error; future actuals are never used here.
        self.use_forecast = bool(use_forecast)
        self.use_tariff_lookahead = bool(use_tariff_lookahead)
        self.base_obs_dim = OBS_DIM_FC if self.use_forecast else OBS_DIM
        self.extra_obs_dim = int(extra_obs_dim)
        self.obs_dim = self.base_obs_dim + self.extra_obs_dim
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
        self._obs_static: np.ndarray | None = None
        self._refresh_cached_coefficients()

    def _refresh_cached_coefficients(self) -> None:
        cfg = self.cfg
        self._inv_p_ref = 1.0 / self.p_ref
        self._inv_peak_price = 1.0 / cfg.price_peak
        self._soc_charge_coeff = self.dt * cfg.eta_ch / cfg.E_cap
        self._soc_discharge_coeff = self.dt / (cfg.eta_dis * cfg.E_cap)
        self._available_power_coeff = cfg.E_cap * cfg.eta_dis / self.dt
        self._room_power_coeff = cfg.E_cap / (cfg.eta_ch * self.dt)
        self._phi_coef = cfg.E_cap * cfg.eta_dis * cfg.price_mid

    def _reset_demand_window(self) -> None:
        self._roll_grid = np.zeros(self.roll_k, dtype=np.float64)
        self._roll_nb = np.zeros(self.roll_k, dtype=np.float64)
        self._roll_pos = 0
        self._roll_count = 0
        self._roll_grid_sum = 0.0
        self._roll_nb_sum = 0.0

    # ------------------------------------------------------------------
    def _configure_control_interval(self) -> None:
        native_minutes = self.dt * 60.0
        control_minutes = self.control_dt_minutes
        ratio = control_minutes / native_minutes
        rounded_ratio = round(ratio)
        if (
            not np.isfinite(control_minutes)
            or control_minutes < native_minutes - 1e-9
            or abs(ratio - rounded_ratio) > 1e-9
            or abs(30.0 / control_minutes - round(30.0 / control_minutes)) > 1e-9
            or abs(1440.0 / control_minutes - round(1440.0 / control_minutes)) > 1e-9
        ):
            raise ValueError(
                "control_dt_minutes must be a native-or-coarser multiple "
                "that divides both 30 minutes and 24 hours"
            )
        self.native_steps_per_action = int(rounded_ratio)

    # ------------------------------------------------------------------
    def reset(self, month: MonthData, soc_init: float | None = None,
              static_observation_cache: np.ndarray | None = None) -> np.ndarray:
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
                self._configure_control_interval()
                self._tar_base = tariff_vector(self.cfg)
                from common import TOU_RULES, cfg_no_peak
                self._tar_sun = tariff_vector(cfg_no_peak(self.cfg)) if TOU_RULES.get("sunday_no_peak") else self._tar_base
                self._refresh_cached_coefficients()
        self.day = 0
        self.t = 0
        self.soc = float(soc_init if soc_init is not None else self.cfg.SOC_eod)
        self.d_run = self.d_run_init    # running monthly 30-min peak (kW)
        self.g_prev = 0.0               # previous-step grid import (kW)
        self.d_run_nb = self.d_run_init  # counterfactual no-BESS running peak
        self.g_prev_nb = 0.0
        if self.record_trajectory:
            self.log_grid = [np.zeros(self.n_steps) for _ in month.days]
            self.log_soc = [np.zeros(self.n_steps + 1) for _ in month.days]
            self.log_pbess = [np.zeros(self.n_steps) for _ in month.days]
        else:
            self.log_grid = []
            self.log_soc = []
            self.log_pbess = []
        self._reset_demand_window()
        self._day_fraction = 1.0 / max(1, len(month.days))
        if self.record_trajectory:
            self.log_soc[0][0] = self.soc
        self._set_day_tariff()
        self._make_day_forecast()
        if static_observation_cache is None:
            self._cache_static_observations()
        else:
            expected = (self.n_steps, self.obs_dim)
            if static_observation_cache.shape != expected:
                raise ValueError(
                    "static observation cache has incompatible shape"
                )
            self._obs_static = static_observation_cache
        return self._obs()

    def _set_day_tariff(self):
        from common import is_sunday
        day = self.month.days[self.day]
        self.tariff = self._tar_sun if is_sunday(day) else self._tar_base

    def _make_day_forecast(self):
        if not self.use_forecast:
            return
        day = self.month.days[self.day]
        forecast = getattr(day, "forecast", None)
        expected = (self.n_steps, 4)
        if forecast is None or np.asarray(forecast).shape != expected:
            raise ValueError(
                f"forecast mode requires real causal predictions shaped {expected}"
            )
        if not np.isfinite(forecast).all():
            raise ValueError("forecast predictions contain non-finite values")

    def _cache_static_observations(self):
        """Cache observation fields that cannot change within the day."""
        day = self.month.days[self.day]
        cache = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        inv_ref = self._inv_p_ref
        working = 1.0 if day.day_type == "working" else 0.0
        day_fraction = self.day * self._day_fraction
        hour_steps = self.n_steps // 24
        for t in range(self.n_steps):
            load, pv = day.load[t], day.pv[t]
            eff = max(0.0, load - pv)
            sur = max(0.0, pv - load)
            t_next = min(t + hour_steps, self.n_steps - 1)
            angle = 2.0 * np.pi * t / self.n_steps
            cache[t, 0] = np.sin(angle)
            cache[t, 1] = np.cos(angle)
            cache[t, 2] = eff * inv_ref
            cache[t, 3] = pv * inv_ref
            cache[t, 4] = sur * inv_ref
            cache[t, 5] = 0.0
            cache[t, 6] = self.tariff[t] * self._inv_peak_price
            cache[t, 7] = self.tariff[t_next] * self._inv_peak_price
            cache[t, 8] = 0.0
            cache[t, 9] = 0.0
            cache[t, 10] = working
            cache[t, 11] = day_fraction
            # sin/cos above already encode time cyclically.  Schema v2 uses
            # this formerly redundant linear-time slot for a dynamic signal.
            cache[t, 12] = 0.0 if self.use_tariff_lookahead else t / self.n_steps
            if self.use_forecast:
                cache[t, 13:17] = self._fc_features(t)
        self._obs_static = cache

    def _fc_features(self, t):
        """Real model predictions: next-hour and following-two-hour
        effective load/PV means, already normalized by p_ref."""
        return self.month.days[self.day].forecast[t]

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        t = self.t
        # A fresh array is required because callers retain the current
        # observation until after step() has produced the next one.
        base = self._obs_static[t].copy()
        base[5] = self.soc
        base[8] = self.d_run * self._inv_p_ref
        base[9] = self.g_prev * self._inv_p_ref
        if self.use_tariff_lookahead:
            lookahead = self._tariff_lookahead(self.t, self.soc)
            base[12] = lookahead["precharge_pressure"]
        self._fill_extra_observation(base, t)
        return base

    def _next_cheap_window(self, t: int) -> tuple[int, int, bool]:
        """Return distance, usable length and day-wrap for the next OFF run.

        OFF comes entirely from the user's tariff-window configuration.  This
        intentionally contains no clock literals: changing the UI window also
        changes what the agent sees and how its reward is shaped.
        """
        off = set(int(step) % self.n_steps for step in self.cfg.OFF)
        if not off:
            return self.n_steps, 0, False
        steps_until = 0
        while steps_until < self.n_steps and (t + steps_until) % self.n_steps not in off:
            steps_until += 1
        if steps_until >= self.n_steps:
            return self.n_steps, 0, False
        length = 0
        while length < self.n_steps and (t + steps_until + length) % self.n_steps in off:
            length += 1
        return steps_until, length, t + steps_until >= self.n_steps

    def _tariff_lookahead(self, t: int, soc: float) -> dict[str, float | bool]:
        steps_until, cheap_steps, starts_next_day = self._next_cheap_window(t)
        max_charge_kw = max(0.0, float(self.cfg.P_rated_nominal))
        room_ac_kwh = max(
            0.0,
            (self.cfg.SOC_max - soc) * self.cfg.E_cap / max(self.cfg.eta_ch, 1e-9),
        )
        cheap_capacity_kwh = cheap_steps * self.dt * max_charge_kw
        deficit_kwh = max(0.0, room_ac_kwh - cheap_capacity_kwh)
        full_band_ac_kwh = max(
            1e-9,
            (self.cfg.SOC_max - self.cfg.SOC_min)
            * self.cfg.E_cap
            / max(self.cfg.eta_ch, 1e-9),
        )
        prewindow_capacity_kwh = steps_until * self.dt * max_charge_kw
        pressure = (
            min(1.0, deficit_kwh / max(prewindow_capacity_kwh, 1e-9))
            if cheap_steps and steps_until > 0
            else 0.0
        )
        return {
            "steps_until": float(steps_until),
            "cheap_capacity_ratio": cheap_capacity_kwh / max(room_ac_kwh, 1e-9),
            "deficit_ratio": min(1.0, deficit_kwh / full_band_ac_kwh),
            "precharge_pressure": pressure,
            "starts_next_day": bool(starts_next_day),
        }

    def _fill_extra_observation(self, observation: np.ndarray, t: int) -> None:
        """Subclass hook for code-native controller context fields."""

    # ------------------------------------------------------------------
    def project_action(self, a: float, load: float, pv: float):
        """Safety projection. Returns (d, cg, cp) all >= 0, AC kW.
        d = discharge to load, cg = charge from grid, cp = charge from
        free PV surplus. Guarantees grid = eff + cg - d >= 0 and SOC in
        [SOC_min, SOC_max] after the step."""
        cfg = self.cfg
        eff = max(0.0, load - pv)
        sur = max(0.0, pv - load)
        # Hard-binarize the policy output before applying physical limits.
        # Zero maps to +1, so the command is always exactly -1 or +1.
        binary_action = 1.0 if float(a) >= 0.0 else -1.0
        p_des = binary_action * cfg.P_rated_nominal
        if p_des >= 0.0:                               # discharge request
            avail = max(0.0, (self.soc - cfg.SOC_min)
                        * self._available_power_coeff)
            d = min(p_des, cfg.P_rated_nominal, eff, avail)
            return d, 0.0, 0.0
        room = max(0.0, (cfg.SOC_max - self.soc)
                   * self._room_power_coeff)
        ch = min(-p_des, cfg.P_rated_nominal, room)
        cp = min(ch, sur)                              # free PV first
        # peak-safe: grid-charging may never set a new monthly peak itself
        cg_cap = max(0.0, self.d_run - eff)
        cg = min(ch - cp, cg_cap)
        return 0.0, cg, cp

    # ------------------------------------------------------------------
    def _step_native(self, action: float):
        cfg = self.cfg
        day = self.month.days[self.day]
        t = self.t
        load, pv = float(day.load[t]), float(day.pv[t])
        eff = max(0.0, load - pv)
        lookahead = self._tariff_lookahead(t, self.soc)

        d, cg, cp = self.project_action(action, load, pv)
        self._last_p_bess_kw = d - (cg + cp)
        grid = eff + cg - d                            # >= 0 by construction
        self.soc = self.soc + (cg + cp) * self._soc_charge_coeff \
                   - d * self._soc_discharge_coeff
        # numerical guard only  projection already bounds SOC
        self.soc = min(cfg.SOC_max, max(cfg.SOC_min, self.soc))

        # --- costs (actual, for logging/scoring) ------------------------
        energy_cost = self.tariff[t] * grid * self.dt
        pos = self._roll_pos
        if self._roll_count == self.roll_k:
            self._roll_grid_sum -= self._roll_grid[pos]
            self._roll_nb_sum -= self._roll_nb[pos]
        else:
            self._roll_count += 1
        self._roll_grid[pos] = grid
        self._roll_nb[pos] = eff
        self._roll_grid_sum += grid
        self._roll_nb_sum += eff
        self._roll_pos = (pos + 1) % self.roll_k
        if self._roll_count < self.roll_k:
            d_t = 0.0                    # cha  ca s 30' u ngy
            d_t_nb = 0.0
        else:
            d_t = self._roll_grid_sum / self.roll_k
            d_t_nb = self._roll_nb_sum / self.roll_k
        peak_pen = cfg.T_cap * max(0.0, d_t - self.d_run)
        self.d_run = max(self.d_run, d_t)
        peak_pen_nb = cfg.T_cap * max(0.0, d_t_nb - self.d_run_nb)
        self.d_run_nb = max(self.d_run_nb, d_t_nb)
        deg_cost = self.deg * (d + cg + cp) * self.dt

        # Learned tariff-window behaviour.  Cheap grid energy gets a dense
        # dopamine signal.  Grid charging outside the user's OFF windows gets
        # a strong stress signal, except when the approaching OFF run cannot
        # physically fill the remaining SOC room at rated power.  In that
        # case urgency smoothly swaps stress for pre-charge dopamine; no
        # action is hard-blocked and no hour is hard-coded.
        grid_charge_kwh = cg * self.dt
        tariff_spread = max(0.0, cfg.price_peak - cfg.price_off)
        cheap_charge_joy = 0.0
        noncheap_charge_stress = 0.0
        precharge_joy = 0.0
        if self.use_tariff_lookahead:
            if t in cfg.OFF:
                cheap_charge_joy = grid_charge_kwh * tariff_spread
            elif grid_charge_kwh > 0.0:
                pressure = float(lookahead["precharge_pressure"])
                noncheap_charge_stress = grid_charge_kwh * (
                    tariff_spread * (1.0 - pressure)
                    + max(0.0, self.tariff[t] - cfg.price_mid)
                )
                precharge_joy = grid_charge_kwh * tariff_spread * pressure

        # --- reward: counterfactual delta vs no-BESS ---------------------
        energy_delta = self.tariff[t] * (cg - d) * self.dt
        peak_delta = peak_pen - peak_pen_nb
        # --- logs ------------------------------------------------------
        if self.record_trajectory:
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
            self._reset_demand_window()
            if self.day >= len(self.month.days):
                done = True
            else:
                if self.record_trajectory:
                    self.log_soc[self.day][0] = self.soc
                self._set_day_tariff()
                self._make_day_forecast()
                self._cache_static_observations()
        obs = None if done else self._obs()
        return (
            obs,
            done,
            grid,
            energy_cost,
            energy_delta,
            peak_delta,
            deg_cost,
            peak_pen,
            peak_pen_nb,
            cheap_charge_joy,
            noncheap_charge_stress,
            precharge_joy,
        )

    def step(self, action: float):
        """Apply one policy decision over one control interval.

        The requested action is held, but projection and all physical/billing
        state updates still run at the dataset's native resolution.
        """
        soc_before = self.soc
        energy_cost_total = 0.0
        energy_delta_total = 0.0
        peak_delta_total = 0.0
        deg_cost_total = 0.0
        peak_pen_total = 0.0
        peak_pen_nb_total = 0.0
        cheap_charge_joy_total = 0.0
        noncheap_charge_stress_total = 0.0
        precharge_joy_total = 0.0
        obs = None
        done = False
        grid = 0.0
        native_rows = 0
        throughput_kwh = 0.0
        abs_p_bess_sum = 0.0
        for _ in range(self.native_steps_per_action):
            (
                obs,
                done,
                grid,
                energy_cost,
                energy_delta,
                peak_delta,
                deg_cost,
                peak_pen,
                peak_pen_nb,
                cheap_charge_joy,
                noncheap_charge_stress,
                precharge_joy,
            ) = self._step_native(action)
            native_rows += 1
            throughput_kwh += abs(self._last_p_bess_kw) * self.dt
            abs_p_bess_sum += abs(self._last_p_bess_kw)
            energy_cost_total += energy_cost
            energy_delta_total += energy_delta
            peak_delta_total += peak_delta
            deg_cost_total += deg_cost
            peak_pen_total += peak_pen
            peak_pen_nb_total += peak_pen_nb
            cheap_charge_joy_total += cheap_charge_joy
            noncheap_charge_stress_total += noncheap_charge_stress
            precharge_joy_total += precharge_joy
            if done:
                break

        phi_before = (soc_before - self.cfg.SOC_min) * self._phi_coef
        phi_after = (self.soc - self.cfg.SOC_min) * self._phi_coef
        shaping = self.gamma * phi_after - phi_before
        reward = (
            -(
                energy_delta_total
                + peak_delta_total
                + deg_cost_total
            )
            + shaping
            + cheap_charge_joy_total
            + precharge_joy_total
            - noncheap_charge_stress_total
        ) / REWARD_SCALE
        return obs, reward, done, {
            "grid_kw": grid,
            "energy_cost": energy_cost_total,
            "energy_delta": energy_delta_total,
            "peak_delta": peak_delta_total,
            "deg_cost": deg_cost_total,
            "peak_pen": peak_pen_total,
            "peak_pen_nb": peak_pen_nb_total,
            "shaping": shaping,
            "cheap_charge_joy": cheap_charge_joy_total,
            "noncheap_charge_stress": noncheap_charge_stress_total,
            "precharge_joy": precharge_joy_total,
            "native_rows": native_rows,
            "d_run": self.d_run,
            "throughput_kwh": throughput_kwh,
            "mean_abs_p_bess_kw": abs_p_bess_sum / max(1, native_rows),
        }
