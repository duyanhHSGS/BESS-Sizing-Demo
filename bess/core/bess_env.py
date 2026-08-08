"""bess.core.bess_env.py  CMDP environment for reactive or real-forecast BESS dispatch.

Design follows the two-layer framework in CoSoLyThuyet_DRL_BESS_Sizing.html
(Hu et al. 2026): the 13-input variant sees real-time measurements only.
The 17-input variant adds four externally prepared, causal look-ahead
predictions. Both output one continuous action in [-1, 1].

HARD-CONSTRAINT SAFETY PROJECTION (never learned, always enforced):
  * zero export : discharge is capped at the net load, so
                  grid[t] = eff_load + cg - d >= 0 for ANY policy output.
  * SOC bounds  : charge/discharge are capped by the energy head-room in
                  [SOC_min, SOC_max] at the current step.
  * P_rated     : |p_bess| <= P_rated on the AC side.
The RL problem is therefore unconstrained for the learner; the projection
makes the 4 critical scenarios in CLAUDE.md structurally satisfiable.

SPARSE DEMAND-CHARGE SHAPING:
  The monthly demand charge T_cap * max_b D_b (D = fixed, clock-aligned
  30-minute meter-block average of grid import) is path-dependent and fires
  once a month. We shape it into a
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

ACTION PROJECTION:
  The environment enforces physical feasibility only: SOC bounds, rated power,
  and zero export. Demand economics are learned from the actual fixed 30-minute
  meter-block reward instead of being hard-coded as a per-sample peak guard.
"""
from __future__ import annotations

import numpy as np

from bess.core.common import tariff_vector, validate_control_interval_minutes
from bess.core.scenario_gen import MonthData
from bess.core.settings import PPO_GAMMA
from bess.core.timebase import demand_window_steps, dt_from_steps_per_day, steps_per_day_from_dt

OBS_DIM = 13
OBS_DIM_FC = 17             # forecast-informed variant (+4 features)
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
                 n_steps: int | None = None,
                 dt_hours: float | None = None,
                 record_trajectory: bool = True,
                 extra_obs_dim: int = 0):
        # dt is the source of truth. It may be 15 min, 1 min, 0.5 min, etc.
        # Step counts are derived from dt instead of hard-coded to 96/day.
        configured_dt = float(getattr(cfg, "dt", 0.0))
        self.dt = float(dt_hours) if dt_hours is not None else configured_dt
        if self.dt <= 0.0:
            raise ValueError("BESS config must provide a positive dt_hours")
        derived_steps_per_day = steps_per_day_from_dt(self.dt)
        if n_steps is not None and int(n_steps) != derived_steps_per_day:
            raise ValueError(
                f"n_steps={n_steps} disagrees with dt_hours={self.dt:g}; "
                f"expected {derived_steps_per_day} steps/day"
            )
        self.n_steps = derived_steps_per_day
        self.control_dt_minutes = (
            float(control_dt_minutes)
            if control_dt_minutes is not None
            else self.dt * 60.0
        )
        self.native_steps_per_action = 1
        cfg.set_dt(self.dt)
        self._configure_control_interval()
        self.roll_k = demand_window_steps(self.dt)
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
        self.base_obs_dim = OBS_DIM_FC if self.use_forecast else OBS_DIM
        self.extra_obs_dim = int(extra_obs_dim)
        self.obs_dim = self.base_obs_dim + self.extra_obs_dim
        normal_day_tariff = tariff_vector(cfg)
        self._tar_base = normal_day_tariff
        # Sunday can use a separate tariff with peak pricing removed.
        from bess.core.common import TOU_RULES, cfg_no_peak
        if TOU_RULES.get("sunday_no_peak"):
            sunday_tariff = tariff_vector(cfg_no_peak(cfg))
            self._tar_sun = sunday_tariff
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
        config = self.cfg
        self._inv_p_ref = 1.0 / self.p_ref
        self._inv_peak_price = 1.0 / config.price_peak
        self._soc_charge_coeff = self.dt * config.eta_ch / config.E_cap
        self._soc_discharge_coeff = self.dt / (config.eta_dis * config.E_cap)
        self._available_power_coeff = config.E_cap * config.eta_dis / self.dt
        self._room_power_coeff = config.E_cap / (config.eta_ch * self.dt)
        # Base stored-energy value used by the SOC shaping reward.
        # The current tariff is multiplied in later inside step().
        self._phi_coef_base = config.E_cap * config.eta_dis

    def _reset_demand_window(self) -> None:
        """Reset the current fixed 30-minute meter integration block."""
        self._block_count = 0
        self._block_grid_sum = 0.0
        self._block_nb_sum = 0.0

    # ------------------------------------------------------------------
    def _configure_control_interval(self) -> None:
        native_step_minutes = self.dt * 60.0
        control_step_minutes = validate_control_interval_minutes(
            native_step_minutes,
            self.control_dt_minutes,
        )
        self.native_steps_per_action = int(round(control_step_minutes / native_step_minutes))

    # ------------------------------------------------------------------
    def reset(self, month: MonthData, soc_init: float | None = None,
              static_observation_cache: np.ndarray | None = None) -> np.ndarray:
        self.month = month
        if month.days:
            data_steps_per_day = len(month.days[0].load)
            if data_steps_per_day <= 0:
                raise ValueError("month contains an empty day")
            if data_steps_per_day != self.n_steps:
                self.n_steps = data_steps_per_day
                self.dt = dt_from_steps_per_day(self.n_steps)
                self.cfg.set_dt(self.dt)
                self.roll_k = demand_window_steps(self.dt)
                self._configure_control_interval()
                self._tar_base = tariff_vector(self.cfg)
                from bess.core.common import TOU_RULES, cfg_no_peak
                self._tar_sun = (
                    tariff_vector(cfg_no_peak(self.cfg))
                    if TOU_RULES.get("sunday_no_peak")
                    else self._tar_base
                )
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
            expected_cache_shape = (self.n_steps, self.obs_dim)
            if static_observation_cache.shape != expected_cache_shape:
                raise ValueError(
                    "static observation cache has incompatible shape"
                )
            self._obs_static = static_observation_cache
        return self._obs()

    def _set_day_tariff(self):
        from bess.core.common import is_sunday
        current_day = self.month.days[self.day]
        self.tariff = self._tar_sun if is_sunday(current_day) else self._tar_base

    def _make_day_forecast(self):
        if not self.use_forecast:
            return
        current_day = self.month.days[self.day]
        forecast_values = getattr(current_day, "forecast", None)
        expected_forecast_shape = (self.n_steps, 4)
        if (
            forecast_values is None
            or np.asarray(forecast_values).shape != expected_forecast_shape
        ):
            raise ValueError(
                "forecast mode requires real causal predictions shaped "
                f"{expected_forecast_shape}"
            )
        if not np.isfinite(forecast_values).all():
            raise ValueError("forecast predictions contain non-finite values")

    def _cache_static_observations(self):
        """Cache observation fields that cannot change within the day."""
        current_day = self.month.days[self.day]
        observation_cache = np.empty((self.n_steps, self.obs_dim), dtype=np.float32)
        inverse_reference_power = self._inv_p_ref
        is_working_day = 1.0 if current_day.day_type == "working" else 0.0
        month_progress = self.day * self._day_fraction
        time_until_tariff_change, time_since_tariff_change = self._tariff_transition_fractions()
        for time_step in range(self.n_steps):
            load_kw = current_day.load[time_step]
            pv_kw = current_day.pv[time_step]
            net_load_kw = max(0.0, load_kw - pv_kw)
            pv_surplus_kw = max(0.0, pv_kw - load_kw)
            time_angle = 2.0 * np.pi * time_step / self.n_steps
            observation_cache[time_step, 0] = np.sin(time_angle)
            observation_cache[time_step, 1] = np.cos(time_angle)
            observation_cache[time_step, 2] = net_load_kw * inverse_reference_power
            observation_cache[time_step, 3] = pv_kw * inverse_reference_power
            observation_cache[time_step, 4] = pv_surplus_kw * inverse_reference_power
            observation_cache[time_step, 5] = 0.0
            observation_cache[time_step, 6] = self.tariff[time_step] * self._inv_peak_price
            # Smooth countdown to the next tariff change (1/n_steps per step
            # remaining), instead of an abrupt one-hour-ahead step: gives the
            # agent an anticipatory ramp toward price drops/rises rather than
            # a signal that only differs from field 6 in the single hour
            # immediately before a transition.
            observation_cache[time_step, 7] = time_until_tariff_change[time_step]
            observation_cache[time_step, 8] = 0.0
            observation_cache[time_step, 9] = 0.0
            observation_cache[time_step, 10] = is_working_day
            observation_cache[time_step, 11] = month_progress
            # Time-since-last-tariff-change, the complementary smooth ramp to
            # field 7. Replaces the old t/n_steps sawtooth, which duplicated
            # the sin/cos time-of-day encoding (fields 0/1) but discontinuously
            # jumped 0.99->0.0 exactly at midnight -- the boundary where the
            # cheap off-peak window begins.
            observation_cache[time_step, 12] = time_since_tariff_change[time_step]
            if self.use_forecast:
                observation_cache[time_step, 13:17] = self._fc_features(time_step)
        self._obs_static = observation_cache

    def _tariff_transition_fractions(self):
        """Per-step (steps-until-next-tariff-change, steps-since-last-change),
        each normalized by n_steps and clipped to [0, 1]."""
        tariff_by_step = self.tariff
        steps_per_day = self.n_steps
        steps_until_change = np.empty(steps_per_day, dtype=np.float64)
        steps_until_change[steps_per_day - 1] = 1.0
        for time_step in range(steps_per_day - 2, -1, -1):
            if tariff_by_step[time_step + 1] != tariff_by_step[time_step]:
                steps_until_change[time_step] = 1.0
            else:
                steps_until_change[time_step] = steps_until_change[time_step + 1] + 1.0
        steps_since_change = np.empty(steps_per_day, dtype=np.float64)
        steps_since_change[0] = 0.0
        for time_step in range(1, steps_per_day):
            if tariff_by_step[time_step] != tariff_by_step[time_step - 1]:
                steps_since_change[time_step] = 0.0
            else:
                steps_since_change[time_step] = steps_since_change[time_step - 1] + 1.0
        time_until_change = np.clip(steps_until_change / steps_per_day, 0.0, 1.0)
        time_since_change = np.clip(steps_since_change / steps_per_day, 0.0, 1.0)
        return time_until_change, time_since_change

    def _fc_features(self, time_step):
        """Real model predictions: next-hour and following-two-hour
        effective load/PV means, already normalized by p_ref."""
        return self.month.days[self.day].forecast[time_step]

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        time_step = self.t
        # A fresh array is required because callers retain the current
        # observation until after step() has produced the next one.
        observation = self._obs_static[time_step].copy()
        observation[5] = self.soc
        observation[8] = self.d_run * self._inv_p_ref
        observation[9] = self.g_prev * self._inv_p_ref
        self._fill_extra_observation(observation, time_step)
        return observation

    def _fill_extra_observation(self, observation: np.ndarray, time_step: int) -> None:
        """Subclass hook for code-native controller context fields."""

    # ------------------------------------------------------------------
    def project_action(self, a: float, load: float, pv: float):
        """Turn the PPO request into safe battery power values.

        Returns ``(discharge_kw, grid_charge_kw, pv_charge_kw)``.
        All three values are non-negative AC kW.
        """
        config = self.cfg
        net_load_kw = max(0.0, load - pv)
        pv_surplus_kw = max(0.0, pv - load)
        requested_battery_kw = (
            float(np.clip(a, -1.0, 1.0)) * config.P_rated_nominal
        )

        if requested_battery_kw >= 0.0:  # positive action = discharge
            available_discharge_kw = max(
                0.0,
                (self.soc - config.SOC_min) * self._available_power_coeff,
            )
            discharge_kw = min(
                requested_battery_kw,
                config.P_rated_nominal,
                net_load_kw,
                available_discharge_kw,
            )
            return discharge_kw, 0.0, 0.0

        available_charge_room_kw = max(
            0.0,
            (config.SOC_max - self.soc) * self._room_power_coeff,
        )
        requested_charge_kw = min(
            -requested_battery_kw,
            config.P_rated_nominal,
            available_charge_room_kw,
        )
        pv_charge_kw = min(requested_charge_kw, pv_surplus_kw)  # free PV first

        # Grid charging is limited only by physical feasibility here. The
        # fixed-block demand charge belongs in the reward/billing model, not in
        # a stricter per-sample rule that can delete valid optimal schedules.
        grid_charge_kw = requested_charge_kw - pv_charge_kw
        return 0.0, grid_charge_kw, pv_charge_kw

    # ------------------------------------------------------------------
    def _step_native(self, action: float):
        """Run one native data step, such as one minute or 15 minutes."""
        config = self.cfg
        current_day = self.month.days[self.day]
        time_step = self.t
        load_kw = float(current_day.load[time_step])
        pv_kw = float(current_day.pv[time_step])
        net_load_kw = max(0.0, load_kw - pv_kw)

        discharge_kw, grid_charge_kw, pv_charge_kw = self.project_action(
            action,
            load_kw,
            pv_kw,
        )
        battery_power_kw = discharge_kw - (grid_charge_kw + pv_charge_kw)
        self._last_p_bess_kw = battery_power_kw
        grid_import_kw = net_load_kw + grid_charge_kw - discharge_kw

        total_charge_kw = grid_charge_kw + pv_charge_kw
        self.soc = (
            self.soc
            + total_charge_kw * self._soc_charge_coeff
            - discharge_kw * self._soc_discharge_coeff
        )
        # Numerical guard only; project_action already keeps SOC in bounds.
        self.soc = min(config.SOC_max, max(config.SOC_min, self.soc))

        # --- actual electricity cost -----------------------------------
        electricity_cost = self.tariff[time_step] * grid_import_kw * self.dt

        # --- fixed 30-minute meter integration block --------------------
        self._block_count += 1
        self._block_grid_sum += grid_import_kw
        self._block_nb_sum += net_load_kw

        demand_peak_penalty = 0.0
        no_bess_peak_penalty = 0.0
        if self._block_count == self.roll_k:
            block_demand_kw = self._block_grid_sum / self.roll_k
            no_bess_block_demand_kw = self._block_nb_sum / self.roll_k

            demand_peak_penalty = config.T_cap * max(
                0.0,
                block_demand_kw - self.d_run,
            )
            self.d_run = max(self.d_run, block_demand_kw)

            no_bess_peak_penalty = config.T_cap * max(
                0.0,
                no_bess_block_demand_kw - self.d_run_nb,
            )
            self.d_run_nb = max(self.d_run_nb, no_bess_block_demand_kw)
            self._reset_demand_window()

        battery_wear_cost = (
            self.deg
            * (discharge_kw + grid_charge_kw + pv_charge_kw)
            * self.dt
        )

        # --- reward pieces: BESS world minus no-BESS world -------------
        battery_energy_cost_delta = (
            self.tariff[time_step] * (grid_charge_kw - discharge_kw) * self.dt
        )
        demand_peak_cost_delta = demand_peak_penalty - no_bess_peak_penalty

        # --- logs -------------------------------------------------------
        if self.record_trajectory:
            self.log_grid[self.day][time_step] = grid_import_kw
            self.log_pbess[self.day][time_step] = battery_power_kw
            self.log_soc[self.day][time_step + 1] = self.soc
        self.g_prev = grid_import_kw
        self.g_prev_nb = net_load_kw

        # --- move time forward -----------------------------------------
        self.t += 1
        episode_done = False
        if self.t >= self.n_steps:
            self.t = 0
            self.day += 1
            self.g_prev = 0.0  # billing windows do not straddle days
            self.g_prev_nb = 0.0
            self._reset_demand_window()
            if self.day >= len(self.month.days):
                episode_done = True
            else:
                if self.record_trajectory:
                    self.log_soc[self.day][0] = self.soc
                self._set_day_tariff()
                self._make_day_forecast()
                self._cache_static_observations()

        next_observation = None if episode_done else self._obs()
        return (
            next_observation,
            episode_done,
            grid_import_kw,
            electricity_cost,
            battery_energy_cost_delta,
            demand_peak_cost_delta,
            battery_wear_cost,
            demand_peak_penalty,
            no_bess_peak_penalty,
        )

    def step(self, action: float):
        """Apply one PPO decision over one control interval.

        The same requested action is held for every native data row inside the
        control interval, while battery physics and billing still update on
        every native row.
        """
        soc_before_action = self.soc
        # Save the tariff seen when PPO made this decision. The native-step loop
        # below may move time forward or even cross midnight.
        decision_tariff = self.tariff[self.t]

        total_electricity_cost = 0.0
        total_battery_energy_cost_delta = 0.0
        total_demand_peak_cost_delta = 0.0
        total_battery_wear_cost = 0.0
        total_demand_peak_penalty = 0.0
        total_no_bess_peak_penalty = 0.0
        next_observation = None
        episode_done = False
        grid_import_kw = 0.0
        native_steps_run = 0
        battery_throughput_kwh = 0.0
        total_absolute_battery_power_kw = 0.0

        for _ in range(self.native_steps_per_action):
            (
                next_observation,
                episode_done,
                grid_import_kw,
                electricity_cost,
                battery_energy_cost_delta,
                demand_peak_cost_delta,
                battery_wear_cost,
                demand_peak_penalty,
                no_bess_peak_penalty,
            ) = self._step_native(action)

            native_steps_run += 1
            battery_throughput_kwh += abs(self._last_p_bess_kw) * self.dt
            total_absolute_battery_power_kw += abs(self._last_p_bess_kw)
            total_electricity_cost += electricity_cost
            total_battery_energy_cost_delta += battery_energy_cost_delta
            total_demand_peak_cost_delta += demand_peak_cost_delta
            total_battery_wear_cost += battery_wear_cost
            total_demand_peak_penalty += demand_peak_penalty
            total_no_bess_peak_penalty += no_bess_peak_penalty

            if episode_done:
                break

        stored_energy_value_per_soc = self._phi_coef_base * decision_tariff
        stored_energy_value_before = (
            soc_before_action - self.cfg.SOC_min
        ) * stored_energy_value_per_soc
        stored_energy_value_after = (
            self.soc - self.cfg.SOC_min
        ) * stored_energy_value_per_soc
        soc_shaping_reward = (
            self.gamma * stored_energy_value_after - stored_energy_value_before
        )

        reward = (
            -(
                total_battery_energy_cost_delta
                + total_demand_peak_cost_delta
                + total_battery_wear_cost
            )
            + soc_shaping_reward
        ) / REWARD_SCALE

        return next_observation, reward, episode_done, {
            # Keep these public info keys unchanged so other code does not break.
            "grid_kw": grid_import_kw,
            "energy_cost": total_electricity_cost,
            "energy_delta": total_battery_energy_cost_delta,
            "peak_delta": total_demand_peak_cost_delta,
            "deg_cost": total_battery_wear_cost,
            "peak_pen": total_demand_peak_penalty,
            "peak_pen_nb": total_no_bess_peak_penalty,
            "shaping": soc_shaping_reward,
            "native_rows": native_steps_run,
            "d_run": self.d_run,
            "throughput_kwh": battery_throughput_kwh,
            "mean_abs_p_bess_kw": (
                total_absolute_battery_power_kw / max(1, native_steps_run)
            ),
        }
