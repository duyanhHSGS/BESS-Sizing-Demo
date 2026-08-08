"""bess.core.bess_env.py  CMDP environment for reactive or real-forecast BESS dispatch.

Design follows the two-layer framework in CoSoLyThuyet_DRL_BESS_Sizing.html
(Hu et al. 2026): the 15-input variant sees real-time measurements and the
state of the current fixed 30-minute demand meter block. The 19-input variant
adds four externally prepared, causal look-ahead
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
  The running monthly peak starts from an explicit shaping floor rather than 0:
  any realistic monthly peak exceeds that floor, so the telescoped sum differs from the
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

REACTIVE_OBSERVATION_DIM = 15
FORECAST_OBSERVATION_DIM = 19             # forecast-informed variant (+4 features)
NORMAL_OBSERVATION_SCHEMA = "bess_meter_aware_v2"
REWARD_SCALE_VND = 1e6          # rewards in millions of VND


def normal_observation_compatibility_error(algo: str, meta: dict) -> str | None:
    """Return why a normal checkpoint cannot use the current BESSEnv contract."""
    if str(algo).lower() == "ppo2":
        return None
    if str(algo).lower() not in {"ppo", "grepo", "grepro", "pro"}:
        return None
    if meta.get("observation_schema") != NORMAL_OBSERVATION_SCHEMA:
        return (
            f"legacy observation schema; expected {NORMAL_OBSERVATION_SCHEMA}. "
            "Retrain before deployment"
        )
    expected = FORECAST_OBSERVATION_DIM if meta.get("obs_variant") == "fc" else REACTIVE_OBSERVATION_DIM
    if meta.get("controller") == "sadrbc_residual":
        expected += 1
    try:
        actual = int(meta.get("obs_dim"))
    except (TypeError, ValueError):
        return "checkpoint is missing a valid obs_dim; retrain before deployment"
    if actual != expected:
        return f"checkpoint obs_dim={actual}, current contract requires {expected}; retrain before deployment"
    try:
        learned_wear = float(meta.get("battery_wear_cost"))
    except (TypeError, ValueError):
        return "checkpoint is missing a valid battery_wear_cost; retrain before deployment"
    if not np.isfinite(learned_wear) or learned_wear < 0.0:
        return "checkpoint battery_wear_cost must be finite and non-negative; retrain before deployment"
    if meta.get("cheap_window_acceptance_passed") is False:
        return "checkpoint failed the cheap-window acceptance gate; retrain before deployment"
    return None


class BESSEnv:
    """Month-long episode using the selected data resolution."""

    def __init__(self, config, reference_power_kw: float = 500.0,
                 degradation_cost_vnd_per_kwh: float | None = None,
                 initial_peak_fraction_of_reference: float = 0.6,
                 initial_running_peak_kw: float | None = None,
                 discount_factor: float = PPO_GAMMA,
                 control_interval_minutes: float | None = None,
                 forecast_enabled: bool = False,
                 steps_per_day: int | None = None,
                 native_timestep_hours: float | None = None,
                 record_trajectory: bool = True,
                 extra_observation_dimensions: int = 0):
        # Native timestep duration is the source of truth. It may represent
        # 15-minute, 1-minute, 0.5-minute, or other day-tiling data.
        configured_timestep_hours = float(getattr(config, "dt", 0.0))
        self.native_timestep_hours = (
            float(native_timestep_hours)
            if native_timestep_hours is not None
            else configured_timestep_hours
        )
        if self.native_timestep_hours <= 0.0:
            raise ValueError("BESS config must provide a positive native timestep in hours")

        derived_steps_per_day = steps_per_day_from_dt(self.native_timestep_hours)
        if steps_per_day is not None and int(steps_per_day) != derived_steps_per_day:
            raise ValueError(
                f"steps_per_day={steps_per_day} disagrees with "
                f"native_timestep_hours={self.native_timestep_hours:g}; "
                f"expected {derived_steps_per_day} steps/day"
            )

        self.steps_per_day = derived_steps_per_day
        self.control_interval_minutes = (
            float(control_interval_minutes)
            if control_interval_minutes is not None
            else self.native_timestep_hours * 60.0
        )
        self.native_samples_per_action = 1
        config.set_dt(self.native_timestep_hours)
        self._configure_action_hold_interval()
        self.samples_per_demand_block = demand_window_steps(self.native_timestep_hours)
        self.config = config
        self.reference_power_kw = float(reference_power_kw)
        resolved_wear_cost = (
            getattr(config, "battery_wear_cost_vnd_per_kwh", None)
            if degradation_cost_vnd_per_kwh is None
            else degradation_cost_vnd_per_kwh
        )
        if resolved_wear_cost is None:
            raise ValueError("BESS config must provide battery wear cost in VND/kWh")
        self.degradation_cost_vnd_per_kwh = float(resolved_wear_cost)
        if not np.isfinite(self.degradation_cost_vnd_per_kwh) or self.degradation_cost_vnd_per_kwh < 0.0:
            raise ValueError("battery wear cost must be finite and >= 0")

        # Start the shaping peak below the site's realistic optimum so the agent
        # actually receives a learning signal when it creates a new monthly peak.
        self.initial_running_peak_kw = (
            float(initial_running_peak_kw)
            if initial_running_peak_kw is not None
            else float(initial_peak_fraction_of_reference) * self.reference_power_kw
        )
        self.discount_factor = float(discount_factor)
        self.record_trajectory = bool(record_trajectory)
        # Forecast mode accepts only externally prepared causal predictions.
        # Missing predictions are an error; future actuals are never used here.
        self.forecast_enabled = bool(forecast_enabled)
        self.base_observation_dimensions = FORECAST_OBSERVATION_DIM if self.forecast_enabled else REACTIVE_OBSERVATION_DIM
        self.extra_observation_dimensions = int(extra_observation_dimensions)
        self.observation_dimensions = self.base_observation_dimensions + self.extra_observation_dimensions
        normal_day_tariff = tariff_vector(config)
        self._normal_day_tariff = normal_day_tariff
        # Sunday can use a separate tariff with peak pricing removed.
        from bess.core.common import TOU_RULES, cfg_no_peak
        if TOU_RULES.get("sunday_no_peak"):
            sunday_tariff = tariff_vector(cfg_no_peak(config))
            self._sunday_tariff = sunday_tariff
        else:
            self._sunday_tariff = self._normal_day_tariff
        self.current_day_tariff = self._normal_day_tariff
        self.month_data: MonthData | None = None
        # trajectory logs (filled during an episode)
        self.grid_import_history: list[np.ndarray] = []
        self.state_of_charge_history: list[np.ndarray] = []
        self.battery_power_history: list[np.ndarray] = []
        self._static_observation_cache: np.ndarray | None = None
        self._refresh_physics_coefficients()

    def _refresh_physics_coefficients(self) -> None:
        config = self.config
        self._inverse_reference_power = 1.0 / self.reference_power_kw
        self._inverse_peak_tariff_price = 1.0 / config.price_peak
        self._state_of_charge_gain_per_charge_kw = self.native_timestep_hours * config.eta_ch / config.E_cap
        self._state_of_charge_loss_per_discharge_kw = self.native_timestep_hours / (config.eta_dis * config.E_cap)
        self._discharge_power_per_soc_fraction = config.E_cap * config.eta_dis / self.native_timestep_hours
        self._charge_power_per_soc_fraction = config.E_cap / (config.eta_ch * self.native_timestep_hours)
        # Base stored-energy value used by the SOC shaping reward.
        # The current tariff is multiplied in later inside step().
        self._deliverable_energy_kwh_per_soc_fraction = config.E_cap * config.eta_dis

    def _reset_demand_meter_block(self) -> None:
        """Reset the current fixed 30-minute meter integration block."""
        self._demand_block_sample_count = 0
        self._demand_block_grid_import_sum_kw = 0.0
        self._no_bess_demand_block_grid_import_sum_kw = 0.0

    # ------------------------------------------------------------------
    def _configure_action_hold_interval(self) -> None:
        native_timestep_minutes = self.native_timestep_hours * 60.0
        validated_control_interval_minutes = validate_control_interval_minutes(
            native_timestep_minutes,
            self.control_interval_minutes,
        )
        self.native_samples_per_action = int(round(validated_control_interval_minutes / native_timestep_minutes))

    # ------------------------------------------------------------------
    def reset(self, month_data: MonthData, initial_state_of_charge: float | None = None,
              static_observation_cache: np.ndarray | None = None) -> np.ndarray:
        self.month_data = month_data
        if month_data.days:
            data_steps_per_day = len(month_data.days[0].load)
            if data_steps_per_day <= 0:
                raise ValueError("month contains an empty day")
            if data_steps_per_day != self.steps_per_day:
                self.steps_per_day = data_steps_per_day
                self.native_timestep_hours = dt_from_steps_per_day(self.steps_per_day)
                self.config.set_dt(self.native_timestep_hours)
                self.samples_per_demand_block = demand_window_steps(self.native_timestep_hours)
                self._configure_action_hold_interval()
                self._normal_day_tariff = tariff_vector(self.config)
                from bess.core.common import TOU_RULES, cfg_no_peak
                self._sunday_tariff = (
                    tariff_vector(cfg_no_peak(self.config))
                    if TOU_RULES.get("sunday_no_peak")
                    else self._normal_day_tariff
                )
                self._refresh_physics_coefficients()
        self.current_day_index = 0
        self.current_timestep_index = 0
        self.state_of_charge = float(initial_state_of_charge if initial_state_of_charge is not None else self.config.SOC_eod)
        self.running_monthly_peak_kw = self.initial_running_peak_kw    # running monthly 30-min peak (kW)
        self.previous_grid_import_kw = 0.0               # previous-step grid import (kW)
        self.no_bess_running_monthly_peak_kw = self.initial_running_peak_kw  # counterfactual no-BESS running peak
        self.previous_no_bess_grid_import_kw = 0.0
        if self.record_trajectory:
            self.grid_import_history = [np.zeros(self.steps_per_day) for _ in month_data.days]
            self.state_of_charge_history = [np.zeros(self.steps_per_day + 1) for _ in month_data.days]
            self.battery_power_history = [np.zeros(self.steps_per_day) for _ in month_data.days]
        else:
            self.grid_import_history = []
            self.state_of_charge_history = []
            self.battery_power_history = []
        self._reset_demand_meter_block()
        self._month_progress_per_day = 1.0 / max(1, len(month_data.days))
        if self.record_trajectory:
            self.state_of_charge_history[0][0] = self.state_of_charge
        self._set_current_day_tariff()
        self._validate_current_day_forecast()
        if static_observation_cache is None:
            self._build_static_observation_cache()
        else:
            expected_cache_shape = (self.steps_per_day, self.observation_dimensions)
            if static_observation_cache.shape != expected_cache_shape:
                raise ValueError(
                    "static observation cache has incompatible shape"
                )
            self._static_observation_cache = static_observation_cache
        return self._build_observation()

    def _set_current_day_tariff(self):
        from bess.core.common import is_sunday
        current_day = self.month_data.days[self.current_day_index]
        self.current_day_tariff = self._sunday_tariff if is_sunday(current_day) else self._normal_day_tariff

    def _validate_current_day_forecast(self):
        if not self.forecast_enabled:
            return
        current_day = self.month_data.days[self.current_day_index]
        forecast_values = getattr(current_day, "forecast", None)
        expected_forecast_shape = (self.steps_per_day, 4)
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

    def _build_static_observation_cache(self):
        """Cache observation fields that cannot change within the day."""
        current_day = self.month_data.days[self.current_day_index]
        observation_cache = np.empty((self.steps_per_day, self.observation_dimensions), dtype=np.float32)
        inverse_reference_power = self._inverse_reference_power
        working_day_flag = 1.0 if current_day.day_type == "working" else 0.0
        month_progress = self.current_day_index * self._month_progress_per_day
        time_until_tariff_change, time_since_tariff_change = self._compute_tariff_transition_fractions()
        for timestep_index in range(self.steps_per_day):
            load_kw = current_day.load[timestep_index]
            pv_kw = current_day.pv[timestep_index]
            net_load_kw = max(0.0, load_kw - pv_kw)
            pv_surplus_kw = max(0.0, pv_kw - load_kw)
            time_of_day_angle = 2.0 * np.pi * timestep_index / self.steps_per_day
            observation_cache[timestep_index, 0] = np.sin(time_of_day_angle)
            observation_cache[timestep_index, 1] = np.cos(time_of_day_angle)
            observation_cache[timestep_index, 2] = net_load_kw * inverse_reference_power
            observation_cache[timestep_index, 3] = pv_kw * inverse_reference_power
            observation_cache[timestep_index, 4] = pv_surplus_kw * inverse_reference_power
            observation_cache[timestep_index, 5] = 0.0
            observation_cache[timestep_index, 6] = self.current_day_tariff[timestep_index] * self._inverse_peak_tariff_price
            # Smooth countdown to the next tariff change (1/steps_per_day per step
            # remaining), instead of an abrupt one-hour-ahead step: gives the
            # agent an anticipatory ramp toward price drops/rises rather than
            # a signal that only differs from field 6 in the single hour
            # immediately before a transition.
            observation_cache[timestep_index, 7] = time_until_tariff_change[timestep_index]
            observation_cache[timestep_index, 8] = 0.0
            observation_cache[timestep_index, 9] = 0.0
            observation_cache[timestep_index, 10] = working_day_flag
            observation_cache[timestep_index, 11] = month_progress
            # Time-since-last-tariff-change, the complementary smooth ramp to
            # field 7. Replaces the old timestep/steps_per_day sawtooth, which duplicated
            # the sin/cos time-of-day encoding (fields 0/1) but discontinuously
            # jumped 0.99->0.0 exactly at midnight -- the boundary where the
            # cheap off-peak window begins.
            observation_cache[timestep_index, 12] = time_since_tariff_change[timestep_index]
            sample_inside_demand_block = timestep_index % self.samples_per_demand_block
            observation_cache[timestep_index, 13] = (
                sample_inside_demand_block / (self.samples_per_demand_block - 1)
                if self.samples_per_demand_block > 1
                else 0.0
            )
            observation_cache[timestep_index, 14] = 0.0
            if self.forecast_enabled:
                observation_cache[timestep_index, 15:19] = self._get_forecast_features(timestep_index)
        self._static_observation_cache = observation_cache

    def _compute_tariff_transition_fractions(self):
        """Per-step (steps-until-next-tariff-change, steps-since-last-change),
        each normalized by steps_per_day and clipped to [0, 1]."""
        tariff_by_timestep = self.current_day_tariff
        steps_per_day = self.steps_per_day
        steps_until_change = np.empty(steps_per_day, dtype=np.float64)
        steps_until_change[steps_per_day - 1] = 1.0
        for timestep_index in range(steps_per_day - 2, -1, -1):
            if tariff_by_timestep[timestep_index + 1] != tariff_by_timestep[timestep_index]:
                steps_until_change[timestep_index] = 1.0
            else:
                steps_until_change[timestep_index] = steps_until_change[timestep_index + 1] + 1.0
        steps_since_change = np.empty(steps_per_day, dtype=np.float64)
        steps_since_change[0] = 0.0
        for timestep_index in range(1, steps_per_day):
            if tariff_by_timestep[timestep_index] != tariff_by_timestep[timestep_index - 1]:
                steps_since_change[timestep_index] = 0.0
            else:
                steps_since_change[timestep_index] = steps_since_change[timestep_index - 1] + 1.0
        time_until_change = np.clip(steps_until_change / steps_per_day, 0.0, 1.0)
        time_since_change = np.clip(steps_since_change / steps_per_day, 0.0, 1.0)
        return time_until_change, time_since_change

    def _get_forecast_features(self, timestep_index):
        """Real model predictions: next-hour and following-two-hour
        effective load/PV means, already normalized by reference_power_kw."""
        return self.month_data.days[self.current_day_index].forecast[timestep_index]

    # ------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        timestep_index = self.current_timestep_index
        # A fresh array is required because callers retain the current
        # observation until after step() has produced the next one.
        observation = self._static_observation_cache[timestep_index].copy()
        observation[5] = self.state_of_charge
        observation[8] = self.running_monthly_peak_kw * self._inverse_reference_power
        observation[9] = self.previous_grid_import_kw * self._inverse_reference_power
        observation[14] = (
            self._demand_block_grid_import_sum_kw
            / self.samples_per_demand_block
            * self._inverse_reference_power
        )
        self._fill_extra_observation_features(observation, timestep_index)
        return observation

    def _fill_extra_observation_features(self, observation: np.ndarray, timestep_index: int) -> None:
        """Subclass hook for code-native controller context fields."""

    # ------------------------------------------------------------------
    def project_action(self, action: float, load_kw: float, pv_kw: float):
        """Turn the requested policy action into safe battery power values.

        Returns ``(discharge_kw, grid_charge_kw, pv_charge_kw)``.
        All three values are non-negative AC kW.
        """
        config = self.config
        net_load_kw = max(0.0, load_kw - pv_kw)
        pv_surplus_kw = max(0.0, pv_kw - load_kw)
        requested_battery_power_kw = (
            float(np.clip(action, -1.0, 1.0)) * config.P_rated_nominal
        )

        if requested_battery_power_kw >= 0.0:  # positive action = discharge
            available_discharge_power_kw = max(
                0.0,
                (self.state_of_charge - config.SOC_min) * self._discharge_power_per_soc_fraction,
            )
            discharge_kw = min(
                requested_battery_power_kw,
                config.P_rated_nominal,
                net_load_kw,
                available_discharge_power_kw,
            )
            return discharge_kw, 0.0, 0.0

        available_charge_power_kw = max(
            0.0,
            (config.SOC_max - self.state_of_charge) * self._charge_power_per_soc_fraction,
        )
        requested_charge_power_kw = min(
            -requested_battery_power_kw,
            config.P_rated_nominal,
            available_charge_power_kw,
        )
        pv_charge_kw = min(requested_charge_power_kw, pv_surplus_kw)  # free PV first

        # Grid charging is limited only by physical feasibility here. The
        # fixed-block demand charge belongs in the reward/billing model, not in
        # a stricter per-sample rule that can delete valid optimal schedules.
        grid_charge_kw = requested_charge_power_kw - pv_charge_kw
        return 0.0, grid_charge_kw, pv_charge_kw

    # ------------------------------------------------------------------
    def _step_native_timestep(self, action: float):
        """Run one native data step, such as one minute or 15 minutes."""
        config = self.config
        current_day = self.month_data.days[self.current_day_index]
        timestep_index = self.current_timestep_index
        load_kw = float(current_day.load[timestep_index])
        pv_kw = float(current_day.pv[timestep_index])
        net_load_kw = max(0.0, load_kw - pv_kw)

        discharge_kw, grid_charge_kw, pv_charge_kw = self.project_action(
            action,
            load_kw,
            pv_kw,
        )
        battery_power_kw = discharge_kw - (grid_charge_kw + pv_charge_kw)
        self._last_battery_power_kw = battery_power_kw
        grid_import_kw = net_load_kw + grid_charge_kw - discharge_kw

        total_charge_kw = grid_charge_kw + pv_charge_kw
        self.state_of_charge = (
            self.state_of_charge
            + total_charge_kw * self._state_of_charge_gain_per_charge_kw
            - discharge_kw * self._state_of_charge_loss_per_discharge_kw
        )
        # Numerical guard only; project_action already keeps SOC in bounds.
        self.state_of_charge = min(config.SOC_max, max(config.SOC_min, self.state_of_charge))

        # --- actual electricity cost -----------------------------------
        electricity_cost_vnd = self.current_day_tariff[timestep_index] * grid_import_kw * self.native_timestep_hours

        # --- fixed 30-minute meter integration block --------------------
        self._demand_block_sample_count += 1
        self._demand_block_grid_import_sum_kw += grid_import_kw
        self._no_bess_demand_block_grid_import_sum_kw += net_load_kw

        demand_peak_penalty_vnd = 0.0
        no_bess_peak_penalty_vnd = 0.0
        if self._demand_block_sample_count == self.samples_per_demand_block:
            block_demand_kw = self._demand_block_grid_import_sum_kw / self.samples_per_demand_block
            no_bess_block_demand_kw = self._no_bess_demand_block_grid_import_sum_kw / self.samples_per_demand_block

            demand_peak_penalty_vnd = config.T_cap * max(
                0.0,
                block_demand_kw - self.running_monthly_peak_kw,
            )
            self.running_monthly_peak_kw = max(self.running_monthly_peak_kw, block_demand_kw)

            no_bess_peak_penalty_vnd = config.T_cap * max(
                0.0,
                no_bess_block_demand_kw - self.no_bess_running_monthly_peak_kw,
            )
            self.no_bess_running_monthly_peak_kw = max(self.no_bess_running_monthly_peak_kw, no_bess_block_demand_kw)
            self._reset_demand_meter_block()

        battery_wear_cost_vnd = (
            self.degradation_cost_vnd_per_kwh
            * (discharge_kw + grid_charge_kw + pv_charge_kw)
            * self.native_timestep_hours
        )

        # --- reward pieces: BESS world minus no-BESS world -------------
        battery_energy_cost_delta_vnd = (
            self.current_day_tariff[timestep_index] * (grid_charge_kw - discharge_kw) * self.native_timestep_hours
        )
        demand_peak_cost_delta_vnd = demand_peak_penalty_vnd - no_bess_peak_penalty_vnd

        # --- logs -------------------------------------------------------
        if self.record_trajectory:
            self.grid_import_history[self.current_day_index][timestep_index] = grid_import_kw
            self.battery_power_history[self.current_day_index][timestep_index] = battery_power_kw
            self.state_of_charge_history[self.current_day_index][timestep_index + 1] = self.state_of_charge
        self.previous_grid_import_kw = grid_import_kw
        self.previous_no_bess_grid_import_kw = net_load_kw

        # --- move time forward -----------------------------------------
        self.current_timestep_index += 1
        episode_done = False
        if self.current_timestep_index >= self.steps_per_day:
            self.current_timestep_index = 0
            self.current_day_index += 1
            self.previous_grid_import_kw = 0.0  # billing windows do not straddle days
            self.previous_no_bess_grid_import_kw = 0.0
            self._reset_demand_meter_block()
            if self.current_day_index >= len(self.month_data.days):
                episode_done = True
            else:
                if self.record_trajectory:
                    self.state_of_charge_history[self.current_day_index][0] = self.state_of_charge
                self._set_current_day_tariff()
                self._validate_current_day_forecast()
                self._build_static_observation_cache()

        next_observation = None if episode_done else self._build_observation()
        return (
            next_observation,
            episode_done,
            grid_import_kw,
            electricity_cost_vnd,
            battery_energy_cost_delta_vnd,
            demand_peak_cost_delta_vnd,
            battery_wear_cost_vnd,
            demand_peak_penalty_vnd,
            no_bess_peak_penalty_vnd,
        )

    def step(self, action: float):
        """Apply one PPO decision over one control interval.

        The same requested action is held for every native data row inside the
        control interval, while battery physics and billing still update on
        every native row.
        """
        state_of_charge_before_action = self.state_of_charge
        # Potential shaping uses the tariff belonging to each state. The
        # native-step loop may cross a tariff or calendar boundary.
        current_state_tariff_vnd_per_kwh = self.current_day_tariff[self.current_timestep_index]

        total_electricity_cost_vnd = 0.0
        total_battery_energy_cost_delta_vnd = 0.0
        total_demand_peak_cost_delta_vnd = 0.0
        total_battery_wear_cost_vnd = 0.0
        total_demand_peak_penalty_vnd = 0.0
        total_no_bess_peak_penalty_vnd = 0.0
        next_observation = None
        episode_done = False
        grid_import_kw = 0.0
        native_samples_processed = 0
        battery_throughput_kwh = 0.0
        sum_absolute_battery_power_kw = 0.0

        for _ in range(self.native_samples_per_action):
            (
                next_observation,
                episode_done,
                grid_import_kw,
                electricity_cost_vnd,
                battery_energy_cost_delta_vnd,
                demand_peak_cost_delta_vnd,
                battery_wear_cost_vnd,
                demand_peak_penalty_vnd,
                no_bess_peak_penalty_vnd,
            ) = self._step_native_timestep(action)

            native_samples_processed += 1
            battery_throughput_kwh += abs(self._last_battery_power_kw) * self.native_timestep_hours
            sum_absolute_battery_power_kw += abs(self._last_battery_power_kw)
            total_electricity_cost_vnd += electricity_cost_vnd
            total_battery_energy_cost_delta_vnd += battery_energy_cost_delta_vnd
            total_demand_peak_cost_delta_vnd += demand_peak_cost_delta_vnd
            total_battery_wear_cost_vnd += battery_wear_cost_vnd
            total_demand_peak_penalty_vnd += demand_peak_penalty_vnd
            total_no_bess_peak_penalty_vnd += no_bess_peak_penalty_vnd

            if episode_done:
                break

        stored_energy_value_vnd_per_soc_fraction = self._deliverable_energy_kwh_per_soc_fraction * current_state_tariff_vnd_per_kwh
        stored_energy_value_before_vnd = (
            state_of_charge_before_action - self.config.SOC_min
        ) * stored_energy_value_vnd_per_soc_fraction
        if episode_done:
            stored_energy_value_after_vnd = 0.0
        else:
            next_state_tariff_vnd_per_kwh = self.current_day_tariff[self.current_timestep_index]
            stored_energy_value_after_vnd = (
                self.state_of_charge - self.config.SOC_min
            ) * self._deliverable_energy_kwh_per_soc_fraction * next_state_tariff_vnd_per_kwh
        state_of_charge_shaping_reward_vnd = (
            self.discount_factor * stored_energy_value_after_vnd - stored_energy_value_before_vnd
        )

        reward = (
            -(
                total_battery_energy_cost_delta_vnd
                + total_demand_peak_cost_delta_vnd
                + total_battery_wear_cost_vnd
            )
            + state_of_charge_shaping_reward_vnd
        ) / REWARD_SCALE_VND

        return next_observation, reward, episode_done, {
            # Keep these public info keys unchanged so other code does not break.
            "grid_kw": grid_import_kw,
            "energy_cost": total_electricity_cost_vnd,
            "energy_delta": total_battery_energy_cost_delta_vnd,
            "peak_delta": total_demand_peak_cost_delta_vnd,
            "deg_cost": total_battery_wear_cost_vnd,
            "peak_pen": total_demand_peak_penalty_vnd,
            "peak_pen_nb": total_no_bess_peak_penalty_vnd,
            "shaping": state_of_charge_shaping_reward_vnd,
            "native_rows": native_samples_processed,
            "d_run": self.running_monthly_peak_kw,
            "throughput_kwh": battery_throughput_kwh,
            "mean_abs_p_bess_kw": (
                sum_absolute_battery_power_kw / max(1, native_samples_processed)
            ),
        }
