"""Waypoint guidance controller and wind-aware point-mass dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ..config import AircraftConfig
from ..core.geodesy import ecef_to_geodetic, ecef_to_enu, enu_basis, geodetic_to_ecef
from ..guidance_config import GuidanceConfig
from .aircraft import AircraftState
from .waypoints import FlightPath

_G_MPS2 = 9.80665
_EPS = 1e-9


def _wrap_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


@dataclass(slots=True)
class GuidanceFeedback:
    effective_mach: float | None = None
    commanded_mach: float | None = None
    source_mach_cutoff: bool = False
    ground_hit_fraction: float | None = None


@dataclass(slots=True)
class GuidanceCommand:
    mode: str
    along_track_distance_m: float
    cross_track_error_m: float
    distance_to_destination_m: float
    desired_heading_deg: float
    desired_bank_deg: float
    desired_pitch_deg: float
    desired_load_factor: float
    desired_speed_mps: float
    desired_mach: float
    desired_body_accel_mps2: np.ndarray
    optimizer_cost: float
    predicted_ground_hit_fraction: float
    predicted_source_cutoff_risk: float
    predicted_effective_mach: float
    optimizer_altitude_adjustment_m: float
    optimizer_mach_adjustment: float


@dataclass(slots=True)
class BoomOptimizationDecision:
    altitude_target_m: float
    mach_target: float
    cost: float
    predicted_ground_hit_fraction: float
    predicted_source_cutoff_risk: float
    predicted_effective_mach: float
    altitude_adjustment_m: float
    mach_adjustment: float


class WaypointGuidanceController:
    """Simple 3D lookahead guidance with boom/mach-cutoff feedback hooks."""

    def __init__(self, flight_path: FlightPath, aircraft_config: AircraftConfig, config: GuidanceConfig):
        self.flight_path = flight_path
        self.aircraft_config = aircraft_config
        self.config = config
        self._mode = "takeoff_climb" if config.mode_switch.enable_takeoff_climb else "enroute"
        self._altitude_bias_m = 0.0
        self._last_effective_mach: float | None = None
        self._effective_mach_ratio: float = 1.0
        self._last_source_cutoff = False
        self._last_ground_hit_fraction: float | None = None
        self._smoothed_ground_hit_fraction: float = 0.0
        self._smoothed_source_cutoff: float = 0.0
        self._optimizer_altitude_adjustment_m: float = 0.0
        self._optimizer_mach_adjustment: float = 0.0

    def apply_feedback(self, feedback: GuidanceFeedback):
        opt_cfg = self.config.boom_avoidance.optimizer
        if feedback.effective_mach is not None:
            self._last_effective_mach = float(feedback.effective_mach)
            if feedback.commanded_mach is not None and feedback.commanded_mach > _EPS:
                ratio = _clip(float(feedback.effective_mach) / float(feedback.commanded_mach), 0.55, 1.55)
                ratio_alpha = _clip(float(opt_cfg.effective_mach_ratio_smoothing), 0.0, 1.0)
                self._effective_mach_ratio = (1.0 - ratio_alpha) * self._effective_mach_ratio + ratio_alpha * ratio
        self._last_source_cutoff = bool(feedback.source_mach_cutoff)
        source_alpha = _clip(float(opt_cfg.source_cutoff_smoothing), 0.0, 1.0)
        source_obs = 1.0 if self._last_source_cutoff else 0.0
        self._smoothed_source_cutoff = (1.0 - source_alpha) * self._smoothed_source_cutoff + source_alpha * source_obs

        boom_cfg = self.config.boom_avoidance
        if feedback.ground_hit_fraction is None:
            self._last_ground_hit_fraction = None
            self._smoothed_ground_hit_fraction = float(opt_cfg.ground_risk_decay) * self._smoothed_ground_hit_fraction
            if self._altitude_bias_m > 0.0:
                self._altitude_bias_m = max(0.0, self._altitude_bias_m - boom_cfg.altitude_bias_decay_m_per_step)
            return
        if not boom_cfg.enable_ground_hit_feedback:
            self._last_ground_hit_fraction = None
            self._smoothed_ground_hit_fraction = 0.0
            return

        hit_fraction = max(0.0, float(feedback.ground_hit_fraction))
        self._last_ground_hit_fraction = hit_fraction
        hit_alpha = _clip(float(opt_cfg.ground_hit_fraction_smoothing), 0.0, 1.0)
        self._smoothed_ground_hit_fraction = (
            (1.0 - hit_alpha) * self._smoothed_ground_hit_fraction + hit_alpha * hit_fraction
        )
        excess = hit_fraction - float(boom_cfg.ground_hit_fraction_threshold)
        if excess > 0.0:
            self._altitude_bias_m = min(
                float(boom_cfg.max_altitude_bias_m),
                self._altitude_bias_m + float(boom_cfg.altitude_gain_m_per_hit_fraction) * excess,
            )
        elif self._altitude_bias_m > 0.0:
            self._altitude_bias_m = max(0.0, self._altitude_bias_m - boom_cfg.altitude_bias_decay_m_per_step)

    def _predict_effective_mach(self, mach: float, altitude_delta_m: float) -> float:
        opt_cfg = self.config.boom_avoidance.optimizer
        alt_term = float(opt_cfg.effective_mach_altitude_gain_per_km) * (altitude_delta_m / 1000.0)
        return float(max(0.2, mach * self._effective_mach_ratio + alt_term))

    def _score_boom_candidate(
        self,
        *,
        candidate_altitude_m: float,
        candidate_mach: float,
        base_route_altitude_m: float,
        base_mach: float,
        state: AircraftState,
        signed_xt_m: float,
        distance_remaining_m: float,
    ) -> BoomOptimizationDecision:
        opt_cfg = self.config.boom_avoidance.optimizer
        mode_cfg = self.config.mode_switch
        horizon_steps = max(1, int(opt_cfg.horizon_steps))

        track_norm = abs(float(signed_xt_m)) / max(float(mode_cfg.abort_cross_track_m), 1.0)
        track_norm = _clip(track_norm, 0.0, 1.0)
        terminal_factor = 1.0 if self._mode == "terminal" else 0.0
        if float(mode_cfg.terminal_distance_m) > 0.0:
            terminal_factor *= _clip(
                1.0 - (distance_remaining_m / float(mode_cfg.terminal_distance_m)),
                0.0,
                1.0,
            )

        altitude_delta_from_state_m = float(candidate_altitude_m - state.alt_m)
        eff_mach = self._predict_effective_mach(candidate_mach, altitude_delta_from_state_m)
        ground_risk = _clip(self._smoothed_ground_hit_fraction, 0.0, 1.0)
        cutoff_risk = _clip(self._smoothed_source_cutoff, 0.0, 1.0)

        ground_sum = 0.0
        cutoff_sum = 0.0
        for step in range(horizon_steps):
            cutoff_deficit = max(0.0, float(opt_cfg.effective_mach_margin) - eff_mach)
            cutoff_obs = cutoff_deficit / max(float(opt_cfg.effective_mach_margin), 1e-6)
            cutoff_risk = _clip(
                float(opt_cfg.ground_risk_decay) * cutoff_risk + (1.0 - float(opt_cfg.ground_risk_decay)) * cutoff_obs,
                0.0,
                1.0,
            )

            mach_excess = max(0.0, candidate_mach - float(self.config.speed.target_mach))
            altitude_relief_km = max(0.0, (candidate_altitude_m - base_route_altitude_m) / 1000.0)
            ground_drive = (
                float(opt_cfg.ground_risk_decay) * ground_risk
                + float(opt_cfg.ground_risk_mach_gain) * mach_excess
                - float(opt_cfg.ground_risk_altitude_relief_per_km) * altitude_relief_km
                + float(opt_cfg.ground_risk_track_gain) * track_norm
                - 0.35 * cutoff_risk
            )
            ground_risk = _clip(ground_drive, 0.0, 1.0)

            ground_sum += ground_risk
            cutoff_sum += cutoff_risk

            # Slight horizon uncertainty drift.
            eff_mach -= 0.0008 * float(step + 1)

        mean_ground_risk = ground_sum / float(horizon_steps)
        mean_cutoff_risk = cutoff_sum / float(horizon_steps)
        altitude_dev_km = abs(candidate_altitude_m - base_route_altitude_m) / 1000.0
        mach_dev = abs(candidate_mach - base_mach)
        terminal_altitude_bias = terminal_factor * max(0.0, candidate_altitude_m - base_route_altitude_m) / 1000.0

        cost = (
            float(opt_cfg.weight_ground_risk) * mean_ground_risk
            + float(opt_cfg.weight_cutoff_risk) * mean_cutoff_risk
            + float(opt_cfg.weight_tracking) * track_norm
            + float(opt_cfg.weight_altitude_deviation) * altitude_dev_km
            + float(opt_cfg.weight_mach_deviation) * mach_dev
            + float(opt_cfg.weight_terminal_altitude_bias) * terminal_altitude_bias
        )

        return BoomOptimizationDecision(
            altitude_target_m=float(candidate_altitude_m),
            mach_target=float(candidate_mach),
            cost=float(cost),
            predicted_ground_hit_fraction=float(mean_ground_risk),
            predicted_source_cutoff_risk=float(mean_cutoff_risk),
            predicted_effective_mach=float(eff_mach),
            altitude_adjustment_m=float(candidate_altitude_m - base_route_altitude_m),
            mach_adjustment=float(candidate_mach - base_mach),
        )

    def _select_boom_aware_targets(
        self,
        *,
        state: AircraftState,
        base_route_altitude_m: float,
        base_mach: float,
        signed_xt_m: float,
        distance_remaining_m: float,
    ) -> BoomOptimizationDecision:
        boom_cfg = self.config.boom_avoidance
        opt_cfg = boom_cfg.optimizer
        if not boom_cfg.enable_ground_hit_feedback or not opt_cfg.enabled:
            eff_mach_pred = self._predict_effective_mach(base_mach, base_route_altitude_m - state.alt_m)
            return BoomOptimizationDecision(
                altitude_target_m=float(base_route_altitude_m),
                mach_target=float(base_mach),
                cost=0.0,
                predicted_ground_hit_fraction=float(_clip(self._smoothed_ground_hit_fraction, 0.0, 1.0)),
                predicted_source_cutoff_risk=float(_clip(self._smoothed_source_cutoff, 0.0, 1.0)),
                predicted_effective_mach=float(eff_mach_pred),
                altitude_adjustment_m=0.0,
                mach_adjustment=0.0,
            )

        constraint_cfg = self.config.constraints
        max_alt_adjust = max(0.0, float(opt_cfg.max_altitude_adjustment_m))
        max_mach_adjust = max(0.0, float(opt_cfg.max_mach_adjustment))
        candidate_altitudes = set()
        candidate_machs = set()

        baseline_alt_with_memory = base_route_altitude_m + self._optimizer_altitude_adjustment_m
        baseline_mach_with_memory = base_mach + self._optimizer_mach_adjustment
        candidate_altitudes.add(float(base_route_altitude_m))
        candidate_altitudes.add(float(baseline_alt_with_memory))
        candidate_machs.add(float(base_mach))
        candidate_machs.add(float(baseline_mach_with_memory))

        for offset in opt_cfg.candidate_altitude_offsets_m:
            raw = base_route_altitude_m + float(offset)
            clipped = _clip(
                raw,
                base_route_altitude_m - max_alt_adjust,
                base_route_altitude_m + max_alt_adjust,
            )
            clipped = _clip(clipped, float(constraint_cfg.min_altitude_m), float(constraint_cfg.max_altitude_m))
            candidate_altitudes.add(float(clipped))
        for offset in opt_cfg.candidate_mach_offsets:
            raw = base_mach + float(offset)
            clipped = _clip(
                raw,
                base_mach - max_mach_adjust,
                base_mach + max_mach_adjust,
            )
            clipped = _clip(clipped, float(self.config.speed.min_mach), float(self.config.speed.max_mach))
            candidate_machs.add(float(clipped))

        best: BoomOptimizationDecision | None = None
        for cand_alt in sorted(candidate_altitudes):
            for cand_mach in sorted(candidate_machs):
                decision = self._score_boom_candidate(
                    candidate_altitude_m=float(cand_alt),
                    candidate_mach=float(cand_mach),
                    base_route_altitude_m=float(base_route_altitude_m),
                    base_mach=float(base_mach),
                    state=state,
                    signed_xt_m=float(signed_xt_m),
                    distance_remaining_m=float(distance_remaining_m),
                )
                if best is None or decision.cost < best.cost:
                    best = decision

        if best is None:
            eff_mach_pred = self._predict_effective_mach(base_mach, base_route_altitude_m - state.alt_m)
            best = BoomOptimizationDecision(
                altitude_target_m=float(base_route_altitude_m),
                mach_target=float(base_mach),
                cost=0.0,
                predicted_ground_hit_fraction=float(_clip(self._smoothed_ground_hit_fraction, 0.0, 1.0)),
                predicted_source_cutoff_risk=float(_clip(self._smoothed_source_cutoff, 0.0, 1.0)),
                predicted_effective_mach=float(eff_mach_pred),
                altitude_adjustment_m=0.0,
                mach_adjustment=0.0,
            )

        target_alt_adjust = _clip(
            best.altitude_target_m - base_route_altitude_m,
            -max_alt_adjust,
            max_alt_adjust,
        )
        target_mach_adjust = _clip(
            best.mach_target - base_mach,
            -max_mach_adjust,
            max_mach_adjust,
        )
        alt_step = max(0.0, float(opt_cfg.max_altitude_step_m_per_cycle))
        mach_step = max(0.0, float(opt_cfg.max_mach_step_per_cycle))
        self._optimizer_altitude_adjustment_m += _clip(
            target_alt_adjust - self._optimizer_altitude_adjustment_m,
            -alt_step,
            alt_step,
        )
        self._optimizer_mach_adjustment += _clip(
            target_mach_adjust - self._optimizer_mach_adjustment,
            -mach_step,
            mach_step,
        )
        self._optimizer_altitude_adjustment_m = _clip(self._optimizer_altitude_adjustment_m, -max_alt_adjust, max_alt_adjust)
        self._optimizer_mach_adjustment = _clip(self._optimizer_mach_adjustment, -max_mach_adjust, max_mach_adjust)

        final_alt = _clip(
            base_route_altitude_m + self._optimizer_altitude_adjustment_m,
            float(constraint_cfg.min_altitude_m),
            float(constraint_cfg.max_altitude_m),
        )
        final_mach = _clip(
            base_mach + self._optimizer_mach_adjustment,
            float(self.config.speed.min_mach),
            float(self.config.speed.max_mach),
        )
        final_effective_mach = self._predict_effective_mach(final_mach, final_alt - state.alt_m)
        return BoomOptimizationDecision(
            altitude_target_m=float(final_alt),
            mach_target=float(final_mach),
            cost=float(best.cost),
            predicted_ground_hit_fraction=float(best.predicted_ground_hit_fraction),
            predicted_source_cutoff_risk=float(best.predicted_source_cutoff_risk),
            predicted_effective_mach=float(final_effective_mach),
            altitude_adjustment_m=float(self._optimizer_altitude_adjustment_m),
            mach_adjustment=float(self._optimizer_mach_adjustment),
        )

    def _signed_cross_track_error_m(self, state: AircraftState, projection: dict) -> float:
        closest_ecef = np.asarray(projection["nearest_ecef_m"], dtype=float)
        closest_to_aircraft_enu = np.asarray(
            ecef_to_enu(state.lat_deg, state.lon_deg, state.alt_m, closest_ecef),
            dtype=float,
        ).reshape(-1)
        tangent_ecef = np.asarray(projection["tangent_ecef"], dtype=float)
        tangent_enu = np.asarray(
            ecef_to_enu(
                state.lat_deg,
                state.lon_deg,
                state.alt_m,
                np.asarray(state.position_ecef_m, dtype=float) + tangent_ecef,
            ),
            dtype=float,
        ).reshape(-1)

        tangent_h = tangent_enu[:2]
        tangent_h_norm = float(np.linalg.norm(tangent_h))
        if tangent_h_norm <= _EPS:
            return float(projection["cross_track_m"])
        tangent_h = tangent_h / tangent_h_norm
        left_normal_h = np.array([-tangent_h[1], tangent_h[0]], dtype=float)
        aircraft_from_path_h = -closest_to_aircraft_enu[:2]
        return float(np.dot(aircraft_from_path_h, left_normal_h))

    def _update_mode(self, state: AircraftState, distance_along_m: float, distance_remaining_m: float, abs_xt_m: float):
        mode_cfg = self.config.mode_switch
        in_takeoff_window = mode_cfg.enable_takeoff_climb and (
            distance_along_m <= mode_cfg.takeoff_distance_m or state.alt_m < mode_cfg.climb_completion_altitude_m
        )
        low_alt_abort = state.alt_m < mode_cfg.abort_min_altitude_m and not in_takeoff_window
        if abs_xt_m >= mode_cfg.abort_cross_track_m or low_alt_abort:
            self._mode = "abort_failsafe"
            return

        if self._mode == "abort_failsafe":
            if abs_xt_m <= mode_cfg.abort_recovery_cross_track_m and state.alt_m >= mode_cfg.abort_min_altitude_m + 150.0:
                self._mode = "enroute"
            else:
                return

        if mode_cfg.enable_terminal and distance_remaining_m <= mode_cfg.terminal_distance_m:
            self._mode = "terminal"
            return

        if in_takeoff_window:
            self._mode = "takeoff_climb"
            return

        self._mode = "enroute"

    def _desired_mach(self) -> float:
        speed_cfg = self.config.speed
        mach = float(speed_cfg.target_mach)
        if self._last_effective_mach is not None:
            mach += float(speed_cfg.effective_mach_gain) * (float(speed_cfg.effective_mach_target) - self._last_effective_mach)

        boom_cfg = self.config.boom_avoidance
        if (not boom_cfg.optimizer.enabled) and self._last_ground_hit_fraction is not None:
            excess = max(0.0, float(self._last_ground_hit_fraction) - float(boom_cfg.ground_hit_fraction_threshold))
            mach -= float(boom_cfg.mach_reduction_per_hit_fraction) * excess
        if self._last_source_cutoff:
            mach += float(boom_cfg.source_cutoff_recovery_mach_step)

        if self._mode == "terminal":
            mach = min(mach, max(float(speed_cfg.min_mach), float(speed_cfg.target_mach) - 0.04))
        elif self._mode == "takeoff_climb":
            mach = max(mach, float(speed_cfg.min_mach) + 0.01)
        elif self._mode == "abort_failsafe":
            mach = max(float(speed_cfg.min_mach) + 0.01, min(mach, float(speed_cfg.target_mach)))

        return _clip(mach, float(speed_cfg.min_mach), float(speed_cfg.max_mach))

    def compute_command(self, state: AircraftState) -> GuidanceCommand:
        projection = self.flight_path.project_ecef(state.position_ecef_m)
        along_track_m = float(projection["distance_m"])
        distance_remaining_m = max(0.0, float(self.flight_path.total_length_m - along_track_m))
        signed_xt_m = self._signed_cross_track_error_m(state, projection)
        abs_xt_m = abs(signed_xt_m)

        self._update_mode(state, along_track_m, distance_remaining_m, abs_xt_m)

        speed_mps = max(float(state.speed_mps), 1.0)
        lat_cfg = self.config.lateral
        lookahead_m = _clip(
            speed_mps * float(lat_cfg.lookahead_time_s),
            float(lat_cfg.min_lookahead_m),
            float(lat_cfg.max_lookahead_m),
        )
        if self._mode == "terminal":
            lookahead_m *= float(lat_cfg.terminal_lookahead_scale)
            lookahead_m = max(lookahead_m, float(lat_cfg.min_lookahead_m))
        elif self._mode == "abort_failsafe":
            lookahead_m = float(lat_cfg.min_lookahead_m)

        target_distance_m = min(float(self.flight_path.total_length_m), along_track_m + lookahead_m)
        target_state = self.flight_path.state_at_distance(target_distance_m)
        target_ecef = np.asarray(target_state["ecef_m"], dtype=float)
        delta_ecef = target_ecef - np.asarray(state.position_ecef_m, dtype=float)
        delta_norm = float(np.linalg.norm(delta_ecef))
        if delta_norm <= 1.0:
            delta_ecef = np.asarray(target_state["tangent_ecef"], dtype=float)
            delta_norm = max(float(np.linalg.norm(delta_ecef)), _EPS)
        delta_ecef = delta_ecef / delta_norm

        desired_track_enu = np.asarray(
            ecef_to_enu(
                state.lat_deg,
                state.lon_deg,
                state.alt_m,
                np.asarray(state.position_ecef_m, dtype=float) + delta_ecef,
            ),
            dtype=float,
        ).reshape(-1)
        desired_track_h = desired_track_enu[:2]
        desired_track_h_norm = float(np.linalg.norm(desired_track_h))
        if desired_track_h_norm <= _EPS:
            desired_track_h = np.array([0.0, 1.0], dtype=float)
            desired_track_h_norm = 1.0
        desired_track_h = desired_track_h / desired_track_h_norm
        desired_heading_rad = float(np.arctan2(desired_track_h[0], desired_track_h[1]))

        velocity_enu = np.asarray(
            ecef_to_enu(
                state.lat_deg,
                state.lon_deg,
                state.alt_m,
                np.asarray(state.position_ecef_m, dtype=float) + np.asarray(state.velocity_ecef_mps, dtype=float),
            ),
            dtype=float,
        ).reshape(-1)
        vh = velocity_enu[:2]
        vh_norm = float(np.linalg.norm(vh))
        if vh_norm <= _EPS:
            current_heading_rad = desired_heading_rad
            current_pitch_rad = 0.0
        else:
            current_heading_rad = float(np.arctan2(vh[0], vh[1]))
            current_pitch_rad = float(np.arctan2(velocity_enu[2], max(vh_norm, _EPS)))

        heading_error_rad = _wrap_pi(desired_heading_rad - current_heading_rad)
        xt_correction_rad = _clip(
            -signed_xt_m * float(lat_cfg.cross_track_gain),
            -np.deg2rad(20.0),
            np.deg2rad(20.0),
        )
        eta_rad = _wrap_pi(heading_error_rad + xt_correction_rad)
        lateral_accel_cmd = 2.0 * speed_mps * speed_mps * np.sin(eta_rad) / max(lookahead_m, 1.0)
        desired_bank_deg = np.rad2deg(np.arctan2(lateral_accel_cmd, _G_MPS2))
        desired_bank_deg = _clip(desired_bank_deg, -self.config.constraints.max_bank_deg, self.config.constraints.max_bank_deg)
        if self._mode == "abort_failsafe":
            desired_bank_deg = 0.0
            lateral_accel_cmd = 0.0
        desired_load_factor = float(1.0 / max(np.cos(np.deg2rad(desired_bank_deg)), 0.2))

        vert_cfg = self.config.vertical
        base_route_alt_target_m = float(target_state["alt_m"] + self._altitude_bias_m)
        if self._mode == "takeoff_climb":
            base_route_alt_target_m = max(base_route_alt_target_m, float(vert_cfg.climb_target_altitude_m))
        if self._mode == "terminal":
            blend_dist = max(float(vert_cfg.terminal_altitude_blend_distance_m), 1.0)
            blend = _clip(distance_remaining_m / blend_dist, 0.0, 1.0)
            dest_alt_m = float(self.flight_path.waypoints[-1].alt_m)
            base_route_alt_target_m = blend * base_route_alt_target_m + (1.0 - blend) * dest_alt_m
        if self._mode == "abort_failsafe":
            base_route_alt_target_m = max(base_route_alt_target_m, float(vert_cfg.climb_target_altitude_m))

        base_route_alt_target_m = _clip(
            base_route_alt_target_m,
            float(self.config.constraints.min_altitude_m),
            float(self.config.constraints.max_altitude_m),
        )
        base_desired_mach = self._desired_mach()
        optimization = self._select_boom_aware_targets(
            state=state,
            base_route_altitude_m=base_route_alt_target_m,
            base_mach=base_desired_mach,
            signed_xt_m=signed_xt_m,
            distance_remaining_m=distance_remaining_m,
        )

        route_alt_target_m = float(optimization.altitude_target_m)
        desired_mach = float(optimization.mach_target)
        alt_error_m = route_alt_target_m - float(state.alt_m)
        vertical_speed_cmd = _clip(
            alt_error_m / max(float(vert_cfg.altitude_time_constant_s), 1.0),
            -float(self.config.constraints.max_descent_rate_mps),
            float(self.config.constraints.max_climb_rate_mps),
        )
        desired_pitch_rad = float(np.arctan2(vertical_speed_cmd, max(speed_mps * 0.8, 1.0)))
        if self._mode == "abort_failsafe":
            desired_pitch_rad = max(desired_pitch_rad, np.deg2rad(5.0))
        desired_pitch_deg = _clip(
            np.rad2deg(desired_pitch_rad),
            -float(self.config.constraints.max_pitch_deg),
            float(self.config.constraints.max_pitch_deg),
        )

        desired_speed_mps = desired_mach * float(self.aircraft_config.reference_sound_speed_mps)
        desired_speed_mps = _clip(
            desired_speed_mps,
            float(self.config.constraints.min_speed_mps),
            float(self.config.constraints.max_speed_mps),
        )

        speed_tau_s = max(float(self.config.speed.speed_time_constant_s), 1.0)
        longitudinal_accel_cmd = _clip(
            (desired_speed_mps - speed_mps) / speed_tau_s,
            -float(self.config.constraints.max_longitudinal_accel_mps2),
            float(self.config.constraints.max_longitudinal_accel_mps2),
        )
        vertical_accel_cmd = _clip(
            (np.deg2rad(desired_pitch_deg) - current_pitch_rad) * speed_mps / max(float(vert_cfg.altitude_time_constant_s), 1.0),
            -_G_MPS2,
            _G_MPS2,
        )
        body_accel = np.array([longitudinal_accel_cmd, lateral_accel_cmd, vertical_accel_cmd], dtype=float)

        return GuidanceCommand(
            mode=self._mode,
            along_track_distance_m=along_track_m,
            cross_track_error_m=signed_xt_m,
            distance_to_destination_m=distance_remaining_m,
            desired_heading_deg=float(np.rad2deg(desired_heading_rad)),
            desired_bank_deg=float(desired_bank_deg),
            desired_pitch_deg=float(desired_pitch_deg),
            desired_load_factor=desired_load_factor,
            desired_speed_mps=float(desired_speed_mps),
            desired_mach=float(desired_speed_mps / max(self.aircraft_config.reference_sound_speed_mps, 1e-6)),
            desired_body_accel_mps2=body_accel,
            optimizer_cost=float(optimization.cost),
            predicted_ground_hit_fraction=float(optimization.predicted_ground_hit_fraction),
            predicted_source_cutoff_risk=float(optimization.predicted_source_cutoff_risk),
            predicted_effective_mach=float(optimization.predicted_effective_mach),
            optimizer_altitude_adjustment_m=float(optimization.altitude_adjustment_m),
            optimizer_mach_adjustment=float(optimization.mach_adjustment),
        )


class GuidedPointMassAircraft:
    """Stateful point-mass aircraft model driven by guidance commands."""

    def __init__(self, flight_path: FlightPath, aircraft_config: AircraftConfig, guidance_config: GuidanceConfig):
        if aircraft_config.reference_sound_speed_mps <= 0.0:
            raise ValueError("reference_sound_speed_mps must be positive")
        if guidance_config.integration_dt_s <= 0.0:
            raise ValueError("guidance.integration_dt_s must be positive")

        self.flight_path = flight_path
        self.aircraft_config = aircraft_config
        self.guidance_config = guidance_config
        self.controller = WaypointGuidanceController(flight_path, aircraft_config, guidance_config)

        first_state = self.flight_path.state_at(self.flight_path.start_time)
        start_time = self.flight_path.start_time
        speed_mps = _clip(
            float(guidance_config.speed.target_mach) * float(aircraft_config.reference_sound_speed_mps),
            float(guidance_config.constraints.min_speed_mps),
            float(guidance_config.constraints.max_speed_mps),
        )
        tangent = np.asarray(first_state["tangent_ecef"], dtype=float)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= _EPS:
            tangent = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            tangent = tangent / tangent_norm
        velocity_ecef = tangent * speed_mps

        start_alt_m = _clip(
            float(first_state["alt_m"]),
            float(guidance_config.constraints.min_altitude_m),
            float(guidance_config.constraints.max_altitude_m),
        )
        position_ecef = geodetic_to_ecef(float(first_state["lat_deg"]), float(first_state["lon_deg"]), start_alt_m)

        self._state = AircraftState(
            time_utc=start_time,
            lat_deg=float(first_state["lat_deg"]),
            lon_deg=float(first_state["lon_deg"]),
            alt_m=float(start_alt_m),
            position_ecef_m=np.asarray(position_ecef, dtype=float),
            velocity_ecef_mps=np.asarray(velocity_ecef, dtype=float),
            speed_mps=float(speed_mps),
            mach=float(speed_mps / aircraft_config.reference_sound_speed_mps),
        )
        self._last_command = self.controller.compute_command(self._state)

    @property
    def start_time(self) -> datetime:
        return self.flight_path.start_time

    @property
    def end_time(self) -> datetime:
        return self.flight_path.end_time

    @property
    def duration_s(self) -> float:
        return float(max(0.0, (self.end_time - self.start_time).total_seconds()))

    def ingest_feedback(self, feedback: GuidanceFeedback):
        self.controller.apply_feedback(feedback)

    def state_at(self, time_utc: datetime, wind_enu_mps: np.ndarray | None = None) -> tuple[AircraftState, GuidanceCommand]:
        target_time = time_utc
        if target_time < self._state.time_utc:
            if target_time <= self.start_time:
                return self._state, self._last_command
            raise ValueError("GuidedPointMassAircraft requires non-decreasing query times")

        wind = np.zeros(3, dtype=float) if wind_enu_mps is None else np.asarray(wind_enu_mps, dtype=float).reshape(-1)
        if wind.size < 3:
            wind = np.pad(wind, (0, 3 - wind.size))
        wind = wind[:3]
        wind[:2] = np.clip(wind[:2], -self.guidance_config.max_wind_mps, self.guidance_config.max_wind_mps)
        wind[2] = 0.0

        remaining_s = float((target_time - self._state.time_utc).total_seconds())
        dt_max = float(self.guidance_config.integration_dt_s)
        while remaining_s > 1e-9:
            dt_s = min(dt_max, remaining_s)
            command = self.controller.compute_command(self._state)
            self._state = self._propagate_step(self._state, command, dt_s, wind)
            self._last_command = command
            remaining_s -= dt_s

        self._last_command = self.controller.compute_command(self._state)
        return self._state, self._last_command

    def _propagate_step(
        self,
        state: AircraftState,
        command: GuidanceCommand,
        dt_s: float,
        wind_enu_mps: np.ndarray,
    ) -> AircraftState:
        pos_ecef = np.asarray(state.position_ecef_m, dtype=float)
        air_vel_ecef = np.asarray(state.velocity_ecef_mps, dtype=float)

        air_vel_enu = np.asarray(
            ecef_to_enu(state.lat_deg, state.lon_deg, state.alt_m, pos_ecef + air_vel_ecef),
            dtype=float,
        ).reshape(-1)
        vh = air_vel_enu[:2]
        vh_norm = float(np.linalg.norm(vh))
        if vh_norm <= _EPS:
            cur_heading_rad = np.deg2rad(command.desired_heading_deg)
            cur_pitch_rad = 0.0
            cur_speed_mps = max(float(state.speed_mps), 1.0)
        else:
            cur_heading_rad = float(np.arctan2(vh[0], vh[1]))
            cur_pitch_rad = float(np.arctan2(air_vel_enu[2], max(vh_norm, _EPS)))
            cur_speed_mps = float(np.linalg.norm(air_vel_enu))

        desired_heading_rad = np.deg2rad(command.desired_heading_deg)
        desired_pitch_rad = np.deg2rad(command.desired_pitch_deg)

        max_dpsi = np.deg2rad(float(self.guidance_config.constraints.max_turn_rate_deg_s) * dt_s)
        heading_step = _clip(_wrap_pi(desired_heading_rad - cur_heading_rad), -max_dpsi, max_dpsi)
        new_heading_rad = cur_heading_rad + heading_step

        max_dtheta = np.deg2rad(float(self.guidance_config.constraints.max_pitch_rate_deg_s) * dt_s)
        pitch_step = _clip(desired_pitch_rad - cur_pitch_rad, -max_dtheta, max_dtheta)
        new_pitch_rad = cur_pitch_rad + pitch_step
        new_pitch_rad = _clip(
            new_pitch_rad,
            -np.deg2rad(float(self.guidance_config.constraints.max_pitch_deg)),
            np.deg2rad(float(self.guidance_config.constraints.max_pitch_deg)),
        )

        max_dv = float(self.guidance_config.constraints.max_longitudinal_accel_mps2) * dt_s
        speed_step = _clip(command.desired_speed_mps - cur_speed_mps, -max_dv, max_dv)
        new_speed_mps = _clip(
            cur_speed_mps + speed_step,
            float(self.guidance_config.constraints.min_speed_mps),
            float(self.guidance_config.constraints.max_speed_mps),
        )

        horiz_speed = new_speed_mps * np.cos(new_pitch_rad)
        new_air_vel_enu = np.array(
            [
                horiz_speed * np.sin(new_heading_rad),
                horiz_speed * np.cos(new_heading_rad),
                new_speed_mps * np.sin(new_pitch_rad),
            ],
            dtype=float,
        )
        new_ground_vel_enu = new_air_vel_enu + np.asarray(wind_enu_mps, dtype=float)

        east, north, up = enu_basis(state.lat_deg, state.lon_deg)
        new_air_vel_ecef = east * new_air_vel_enu[0] + north * new_air_vel_enu[1] + up * new_air_vel_enu[2]
        new_ground_vel_ecef = east * new_ground_vel_enu[0] + north * new_ground_vel_enu[1] + up * new_ground_vel_enu[2]

        new_pos_ecef = pos_ecef + new_ground_vel_ecef * dt_s
        lat_deg, lon_deg, alt_m = ecef_to_geodetic(new_pos_ecef[0], new_pos_ecef[1], new_pos_ecef[2])
        alt_m = _clip(
            alt_m,
            float(self.guidance_config.constraints.min_altitude_m),
            float(self.guidance_config.constraints.max_altitude_m),
        )
        new_pos_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)

        new_time = state.time_utc + timedelta(seconds=float(dt_s))
        return AircraftState(
            time_utc=new_time,
            lat_deg=float(lat_deg),
            lon_deg=float(lon_deg),
            alt_m=float(alt_m),
            position_ecef_m=np.asarray(new_pos_ecef, dtype=float),
            velocity_ecef_mps=np.asarray(new_air_vel_ecef, dtype=float),
            speed_mps=float(new_speed_mps),
            mach=float(new_speed_mps / max(float(self.aircraft_config.reference_sound_speed_mps), 1e-6)),
        )
