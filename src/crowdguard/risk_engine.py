"""Feature engineering and multi-modal risk scoring.

What changed from the naive prototype, and why it matters
--------------------------------------------------------
1. **Density is now local, not global.** `count / declared_area` gives the same
   answer for 60 people spread evenly and 60 people packed into one corner --
   but only the second one kills anybody. Density is now estimated per person
   with a k-nearest-neighbour estimator, rho_i = k / (pi * r_k^2), and the
   scene value reported is a robust high percentile of that distribution.

2. **Density is measured on the ground plane, not the image plane**, whenever a
   homography is available (see `calibration.py`). Persons/m2 from raw pixels is
   not a physical quantity.

3. **Crowd pressure is Helbing's definition**, P = rho_local * Var(v_local),
   rather than an invented product of normalised features. It is the published
   precursor of crowd turbulence, and it has a published critical value.

4. **Temporal structure is measured.** Stop-and-go waves, inflow rate and speed
   surges cannot be seen in a single frame, and they are exactly what
   distinguishes a dangerous crowd from a merely busy one.

5. **The output says what kind of risk it is** (see `risk_taxonomy.py`), because
   a crush and a panic dispersal need opposite interventions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import GroundPlane, GroundPlaneCalibrator
from .config import CalibrationConfig, RiskConfig
from .risk_taxonomy import RiskTypeClassifier, TypologyResult
from .zones import ZoneManager, ZoneReading

# Two people cannot occupy the same point; this floor keeps the KNN estimator
# finite when detections overlap, and corresponds to roughly shoulder width.
MIN_SEPARATION_M = 0.30
# Physical ceiling on human packing density (persons/m2).
MAX_PHYSICAL_DENSITY = 10.0


@dataclass
class SceneFeatures:
    """One frame of measurement. Field names kept stable for the JSONL log."""

    frame_id: int
    timestamp_sec: float
    person_count: int

    # density
    density: float                     # mean/global density (legacy field)
    local_density_peak: float          # robust peak local density, persons/m2
    local_density_mean: float

    # kinematics
    avg_speed: float                   # normalised by frame diagonal (legacy)
    avg_speed_ms: float                # metres per second
    flow_disorder: float
    counterflow_index: float           # circular bimodality: opposing streams
    bottleneck_ratio: float            # chokepoint load: density / design capacity
    bottleneck_concentration: float    # how constricted the loaded zone actually is
    lateral_spread: float              # width of the pile-up vs corridor width

    # physics
    crowd_pressure: float              # Helbing: rho_local * Var(v), s^-2
    flux: float                        # J = rho * v, persons per metre per second
    flux_efficiency: float             # J / running peak J

    # temporal
    oscillation_index: float
    density_rate_per_min: float
    count_rate_per_min: float
    speed_surge_ratio: float
    sustained_density_sec: float       # time held above the hard density limit

    # verdict
    risk_score: float
    risk_level: str
    factors: List[str]

    # typology
    primary_risk_type: str = "NORMAL_FLOW"
    primary_risk_label: str = "Normal Flow"
    primary_risk_score: float = 0.0
    hazard_index: float = 0.0
    risk_type_ranking: List[Dict[str, Any]] = field(default_factory=list)
    type_evidence: List[str] = field(default_factory=list)

    # spatial
    zones: List[Dict[str, Any]] = field(default_factory=list)
    hotspot_zone: str = ""
    hotspot_density: float = 0.0

    # plain-language change over the last minute -- the statement a human
    # watching the same screen cannot produce
    trend_summary: str = ""

    # Routing for the hotspot zone: where to divert, where to meter, where to
    # stage marshals. Carried through so the advisory can name a destination
    # instead of only naming the problem.
    hotspot_alternate: str = ""
    hotspot_meter_at: str = ""
    hotspot_staging: str = ""

    # provenance
    contributions: Dict[str, float] = field(default_factory=dict)
    calibrated: bool = False
    calibration_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def typology(self) -> str:
        return self.primary_risk_label


class RiskEngine:
    """Stateful risk engine. One instance per camera / stream."""

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        calibration: Optional[CalibrationConfig] = None,
        zone_manager: Optional[ZoneManager] = None,
    ):
        self.config = config or RiskConfig()
        self.calibration_config = calibration or CalibrationConfig(
            fallback_area_m2=self.config.camera_area_m2
        )
        self.zone_manager = zone_manager
        self.classifier = RiskTypeClassifier(
            pressure_warning=self.config.pressure_warning,
            pressure_critical=self.config.pressure_critical,
        )

        self.ground: Optional[GroundPlane] = None
        w = self.config.temporal_window
        self._speed_hist: Deque[float] = deque(maxlen=w)
        self._density_hist: Deque[float] = deque(maxlen=w)
        self._count_hist: Deque[float] = deque(maxlen=w)
        self._time_hist: Deque[float] = deque(maxlen=w)
        self._flux_hist: Deque[float] = deque(maxlen=w * 4)
        self._peak_flux: float = 1e-6
        self._sustained_density_sec: float = 0.0
        self._last_time: Optional[float] = None
        # Long-run travel axis and speed baseline. Both are properties of the
        # SITE, not of the moment, so they use a slow exponential average rather
        # than the short analysis window.
        self._travel_axis = np.zeros(2, dtype=np.float64)
        self._speed_baseline: float = 0.0
        # A separate, much longer history than the analysis window. The
        # operator-facing statement is "density rose from 2.8 to 4.1 over the
        # last minute", and a seven-second window cannot say that. This is the
        # single most useful sentence the system produces, because it is the
        # comparison a human watching the same screen cannot make.
        self._long_hist: Deque[Tuple[float, float, float, int]] = deque(maxlen=2000)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def ensure_ground_plane(self, frame_shape: Tuple[int, ...]) -> GroundPlane:
        if self.ground is None or self.ground.frame_shape != (int(frame_shape[0]), int(frame_shape[1])):
            self.calibration_config.fallback_area_m2 = self.config.camera_area_m2
            self.ground = GroundPlaneCalibrator(self.calibration_config).build(frame_shape)
        return self.ground

    def reset_history(self) -> None:
        for d in (self._speed_hist, self._density_hist, self._count_hist, self._time_hist, self._flux_hist):
            d.clear()
        self._peak_flux = 1e-6
        self._sustained_density_sec = 0.0
        self._last_time = None
        self._travel_axis = np.zeros(2, dtype=np.float64)
        self._speed_baseline = 0.0
        self._long_hist.clear()

    @staticmethod
    def _clip01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    # ------------------------------------------------------------------ #
    # spatial estimators
    # ------------------------------------------------------------------ #
    def local_densities(self, world_pts: np.ndarray) -> np.ndarray:
        """kNN density estimate per person, in persons per square metre."""
        n = world_pts.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.float64)
        if n == 1:
            return np.array([1.0 / max(1.0, self.config.camera_area_m2)], dtype=np.float64)

        k = int(min(self.config.knn_k, n - 1))
        d = np.linalg.norm(world_pts[:, None, :] - world_pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        d.sort(axis=1)
        r_k = np.maximum(d[:, k - 1], MIN_SEPARATION_M)
        dens = k / (np.pi * r_k ** 2)
        return np.clip(dens, 0.0, MAX_PHYSICAL_DENSITY)

    def local_pressures(self, world_pts: np.ndarray, vel_ms: np.ndarray, dens: np.ndarray) -> np.ndarray:
        """Helbing crowd pressure per person: rho_local * Var(v_local)."""
        n = world_pts.shape[0]
        if n == 0 or vel_ms.shape[0] != n:
            return np.zeros((max(0, n),), dtype=np.float64)
        if n == 1:
            return np.zeros((1,), dtype=np.float64)

        r = float(self.config.neighbourhood_radius_m)
        d = np.linalg.norm(world_pts[:, None, :] - world_pts[None, :, :], axis=-1)
        neighbour = d <= r
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            idx = neighbour[i]
            if idx.sum() < 2:
                continue
            v = vel_ms[idx]
            var = float(np.mean(np.sum((v - v.mean(axis=0)) ** 2, axis=1)))
            out[i] = dens[i] * var
        return out

    def _counterflow_index(self, displacements: np.ndarray) -> float:
        """Detect genuinely opposing streams from where people have actually got to.

        Directional disorder cannot do this job. A stalled crowd shuffles in
        scattered micro-directions and scores just as "disordered" as two
        streams walking into each other, and the two need opposite
        interventions -- separate the streams, versus never reverse a jammed
        crowd -- so conflating them is a safety failure, not a cosmetic one.

        Nor can instantaneous velocity. At a funnel, people fan inward from
        both sides while still walking forward, so their velocity directions
        split into two opposing lateral modes that look exactly like
        counter-flow even though nobody is walking backwards.

        Net displacement over each track's history settles both cases. Someone
        converging on a gate has still ended up further down the corridor;
        someone in a genuine counter-flow has ended up behind the people coming
        the other way. Two circular statistics over those displacement
        directions then separate the regimes:

            R1 = |mean(exp(i.theta))|    one dominant direction
            R2 = |mean(exp(2i.theta))|   one dominant AXIS

        One stream: R1 and R2 both high. Scattered jam: both low. Two opposing
        streams share an axis but cancel in direction -- R2 high, R1 low --
        which is what ``R2 * (1 - R1)`` isolates.
        """
        if displacements.shape[0] < 6:
            return 0.0
        mag = np.linalg.norm(displacements, axis=1)
        # People who have not actually gone anywhere carry no direction
        # information; including them just adds noise from tracker jitter.
        travelled = mag > 0.35
        if travelled.sum() < 6:
            return 0.0
        d = displacements[travelled]
        m = mag[travelled]
        w = m / (m.sum() + 1e-9)
        theta = np.arctan2(d[:, 1], d[:, 0])
        r1 = float(abs(np.sum(w * np.exp(1j * theta))))
        r2 = float(abs(np.sum(w * np.exp(2j * theta))))
        return float(np.clip(r2 * (1.0 - r1), 0.0, 1.0))

    def _bottleneck_ratio(self, image_pts: np.ndarray, frame_shape: Tuple[int, ...],
                          zone_readings: Sequence[ZoneReading]) -> float:
        """Share of the crowd sitting in the most loaded constriction.

        Prefers operator-defined gate/exit zones; falls back to the central
        image region only when no zones are configured.
        """
        # A bottleneck is a *constriction under load*, so the measurement is
        # how close the chokepoint is to its own design capacity -- not how
        # many people happen to be standing in a large zone. Occupancy share
        # is useless here: a zone covering half the frame holds half the crowd
        # when nothing at all is wrong.
        gates = [z for z in zone_readings if z.kind in {"gate", "exit"}]
        if gates:
            return float(np.clip(max(z.load_factor for z in gates), 0.0, 1.0))

        corridors = [z for z in zone_readings if z.kind == "corridor"]
        if corridors:
            peak = max(z.concentration for z in corridors)
            return float(np.clip((peak - 1.0) / 2.0, 0.0, 1.0))

        if image_pts.shape[0] == 0:
            return 0.0
        h, w = float(frame_shape[0]), float(frame_shape[1])
        in_zone = (
            (image_pts[:, 0] >= 0.30 * w) & (image_pts[:, 0] <= 0.70 * w)
            & (image_pts[:, 1] >= 0.20 * h) & (image_pts[:, 1] <= 0.85 * h)
        )
        return float(in_zone.sum() / max(1, image_pts.shape[0]))

    # ------------------------------------------------------------------ #
    # temporal estimators
    # ------------------------------------------------------------------ #
    def _oscillation_index(self) -> float:
        """Stop-and-go wave detector.

        Counts sign reversals in the mean-speed derivative and scales them by
        the relative amplitude of the swing, so a steadily accelerating crowd
        scores 0 while a crowd surging and stalling repeatedly scores high.
        """
        if len(self._speed_hist) < 5:
            return 0.0
        s = np.asarray(self._speed_hist, dtype=np.float64)
        diffs = np.diff(s)
        if diffs.size < 2:
            return 0.0
        signs = np.sign(diffs)
        nonzero = signs[signs != 0]
        if nonzero.size < 2:
            return 0.0
        reversals = float(np.sum(nonzero[1:] != nonzero[:-1]))
        reversal_rate = reversals / max(1.0, nonzero.size - 1)
        mean_s = float(np.mean(s))
        amplitude = float(np.std(s) / mean_s) if mean_s > 1e-6 else 0.0
        scale = max(1e-3, float(getattr(self.config, "oscillation_amplitude_scale", 0.18)))
        return self._clip01(reversal_rate * self._clip01(amplitude / scale))

    def _rate_per_min(self, history: Deque[float]) -> float:
        """Least-squares slope of a series against its timestamps, per minute.

        The series is exponentially smoothed first. Fitting a slope to the raw
        per-frame measurement amplifies noise enormously: a crowd of thirty
        people standing still produced an apparent density trend of nearly two
        persons per square metre per minute, purely from detector flicker.
        """
        if len(history) < 6 or len(self._time_hist) < 6:
            return 0.0
        n = min(len(history), len(self._time_hist))
        raw = np.asarray(list(history)[-n:], dtype=np.float64)
        alpha = 0.25
        y = np.empty_like(raw)
        y[0] = raw[0]
        for i in range(1, raw.size):
            y[i] = alpha * raw[i] + (1 - alpha) * y[i - 1]
        t = np.asarray(list(self._time_hist)[-n:], dtype=np.float64)
        span = t[-1] - t[0]
        if span <= 1e-6:
            return 0.0
        t = t - t[0]
        slope = float(np.polyfit(t, y, 1)[0])
        return slope * 60.0

    def _speed_surge_ratio(self, current: float) -> float:
        """How far above its long-run baseline the crowd is currently moving."""
        if len(self._speed_hist) < 6 or self._speed_baseline < 1e-3:
            return 1.0
        return float(current / self._speed_baseline)

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def compute_features(
        self,
        frame_id: int,
        timestamp_sec: float,
        frame_shape: Tuple[int, ...],
        detections: List[Dict[str, Any]],
        tracks: Dict[int, Tuple[int, int]],
        velocity_vectors: Dict[int, Tuple[float, float]],
        dt_sec: Optional[float] = None,
        active_tracks: Optional[Dict[int, Tuple[int, int]]] = None,
        track_ages: Optional[Dict[int, int]] = None,
        displacements: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> SceneFeatures:
        ground = self.ensure_ground_plane(frame_shape)
        cfg = self.config

        count = len(detections)
        density_global = count / max(1.0, cfg.camera_area_m2)

        # --- align tracks and velocities into matrices --------------------
        # `tracks` may include coasting tracks that were not observed this
        # frame; they all sit on their last known coordinate and would fabricate
        # density. Spatial statistics use observed tracks only.
        if active_tracks is not None:
            tracks = {k: v for k, v in tracks.items() if k in active_tracks}
        track_ids = list(tracks.keys())
        if track_ids:
            image_pts = np.asarray([tracks[i] for i in track_ids], dtype=np.float64)
            vel_px = np.asarray(
                [velocity_vectors.get(i, (0.0, 0.0)) for i in track_ids], dtype=np.float64
            )
        else:
            image_pts = np.zeros((0, 2), dtype=np.float64)
            vel_px = np.zeros((0, 2), dtype=np.float64)

        world_pts = ground.to_world(image_pts) if image_pts.size else np.zeros((0, 2))

        # --- velocity in metres per second --------------------------------
        if dt_sec is None or dt_sec <= 1e-6:
            dt_sec = timestamp_sec - self._time_hist[-1] if self._time_hist else 0.04
            dt_sec = float(dt_sec) if dt_sec and dt_sec > 1e-6 else 0.04
        if world_pts.shape[0]:
            world_next = ground.to_world(image_pts + vel_px)
            vel_ms = (world_next - world_pts) / dt_sec
            # Reject physically impossible speeds from ID switches (>8 m/s).
            speeds_raw = np.linalg.norm(vel_ms, axis=1)
            bad = speeds_raw > 8.0
            if bad.any():
                vel_ms[bad] = vel_ms[bad] / speeds_raw[bad, None] * 8.0
        else:
            vel_ms = np.zeros((0, 2), dtype=np.float64)

        # A track observed once or twice has no meaningful velocity: its
        # apparent direction is detector noise, and noise directions cancel,
        # which registers as counter-flow. Excluding young tracks from the
        # kinematic statistics removes a large source of false disorder.
        min_age = max(2, int(getattr(self.config, "min_track_age", 3)))
        if track_ages is not None and track_ids:
            mature = np.asarray([track_ages.get(i, 0) >= min_age for i in track_ids], dtype=bool)
        else:
            mature = np.ones((len(track_ids),), dtype=bool)

        vel_ms_mature = vel_ms[mature] if vel_ms.size else vel_ms
        speeds_ms = np.linalg.norm(vel_ms_mature, axis=1) if vel_ms_mature.size else np.zeros((0,))
        avg_speed_ms = float(speeds_ms.mean()) if speeds_ms.size else 0.0

        # legacy normalised speed, kept so old logs stay comparable
        diag = float(np.sqrt(frame_shape[0] ** 2 + frame_shape[1] ** 2)) or 1.0
        avg_speed_norm = (
            float(np.mean(np.linalg.norm(vel_px, axis=1)) / diag) if vel_px.size else 0.0
        )

        # --- local density and pressure ------------------------------------
        per_density = self.local_densities(world_pts)
        if per_density.size:
            local_density_peak = float(np.percentile(per_density, cfg.density_percentile))
            local_density_mean = float(per_density.mean())
        else:
            local_density_peak = local_density_mean = 0.0

        per_pressure = self.local_pressures(world_pts, vel_ms, per_density)
        crowd_pressure = (
            float(np.percentile(per_pressure, cfg.density_percentile)) if per_pressure.size else 0.0
        )

        # --- flow disorder --------------------------------------------------
        disorder = 0.0
        if speeds_ms.size >= 4:
            # Only genuinely moving, mature tracks contribute. Below a handful
            # of movers the statistic is meaningless.
            moving = speeds_ms > 0.05
            if moving.sum() >= 4:
                resultant = float(np.linalg.norm(vel_ms_mature[moving].sum(axis=0)))
                disorder = self._clip01(1.0 - resultant / (speeds_ms[moving].sum() + 1e-9))

        if displacements is not None and track_ids:
            disp_px = np.asarray(
                [displacements.get(i, (0.0, 0.0)) for i in track_ids], dtype=np.float64
            )
            disp_world = ground.to_world(image_pts + disp_px) - world_pts if world_pts.size else disp_px
            counterflow_index = self._counterflow_index(disp_world[mature])
        else:
            counterflow_index = 0.0
        if vel_ms_mature.size:
            mean_vec = vel_ms_mature.mean(axis=0)
            self._travel_axis = 0.985 * self._travel_axis + 0.015 * mean_vec
        # Long-run speed baseline for surge detection. A short window is
        # useless here: during a sustained flight the short-window baseline
        # rises to match, and the surge disappears from view within seconds --
        # exactly when the operator most needs to see it.
        if avg_speed_ms > 0:
            self._speed_baseline = (
                avg_speed_ms if self._speed_baseline <= 0
                else 0.99 * self._speed_baseline + 0.01 * avg_speed_ms
            )

        self._long_hist.append((float(timestamp_sec), float(local_density_peak),
                                float(avg_speed_ms), int(count)))

        # --- sustained compression -------------------------------------------
        # A crush is *progressive*: it is built by density held above the safe
        # band for a period, not by one frame touching it. A surge is the
        # opposite -- brief, oscillating. Time-above-threshold separates them.
        if self._last_time is not None:
            elapsed = max(0.0, float(timestamp_sec) - self._last_time)
            if local_density_peak >= cfg.density_hard_limit:
                self._sustained_density_sec += elapsed
            else:
                self._sustained_density_sec = max(0.0, self._sustained_density_sec - elapsed * 2.0)
        self._last_time = float(timestamp_sec)

        # --- zones -----------------------------------------------------------
        zone_readings: List[ZoneReading] = []
        if self.zone_manager is not None and self.zone_manager.enabled and len(self.zone_manager):
            zone_readings = self.zone_manager.readings(
                image_pts, vel_ms, per_density, per_pressure, frame_shape, cfg.density_percentile
            )

        bottleneck = self._bottleneck_ratio(image_pts, frame_shape, zone_readings)
        # A loaded zone is only a bottleneck if it is genuinely a constriction.
        # A full-width closure loads a zone just as hard while concentrating
        # nobody, and it needs the opposite response -- clear the obstruction,
        # not open more lanes.
        # Lateral spread of the loaded region, as a fraction of the corridor
        # width. A narrow aperture squeezes people into a slim column; a
        # full-width closure stops a broad front. Both load a zone identically,
        # and they need opposite responses -- open more lanes, versus clear the
        # obstruction -- so the shape of the pile-up has to be measured.
        lateral_spread = 1.0
        if world_pts.shape[0] >= 8 and per_density.size:
            hot = per_density >= np.percentile(per_density, 70)
            if hot.sum() >= 5:
                xs = world_pts[hot, 0]
                span = float(np.percentile(xs, 90) - np.percentile(xs, 10))
                full = float(np.percentile(world_pts[:, 0], 95) - np.percentile(world_pts[:, 0], 5))
                lateral_spread = float(np.clip(span / max(0.5, full), 0.0, 1.0))

        gate_zones = [z for z in zone_readings if z.kind in {"gate", "exit"}]
        bottleneck_concentration = (
            float(max(z.concentration for z in gate_zones)) if gate_zones else 1.0
        )

        # --- flux and the fundamental diagram ---------------------------------
        flux = float(local_density_mean * avg_speed_ms)
        self._flux_hist.append(flux)
        if flux > self._peak_flux:
            self._peak_flux = flux
        else:
            # Slow decay so a single early peak does not permanently suppress
            # the efficiency signal across a long monitoring session.
            self._peak_flux = max(1e-6, self._peak_flux * 0.9995)
        flux_efficiency = self._clip01(flux / self._peak_flux) if self._peak_flux > 1e-6 else 1.0

        # --- temporal ---------------------------------------------------------
        self._time_hist.append(float(timestamp_sec))
        self._speed_hist.append(avg_speed_ms)
        self._density_hist.append(local_density_peak)
        self._count_hist.append(float(count))

        oscillation = self._oscillation_index()
        density_rate = self._rate_per_min(self._density_hist)
        count_rate = self._rate_per_min(self._count_hist)
        surge = self._speed_surge_ratio(avg_speed_ms)

        # --- normalisation ------------------------------------------------------
        density_norm = self._clip01(
            (local_density_peak - 0.5) / max(0.1, cfg.density_hard_limit - 0.5)
        )
        pressure_norm = self._clip01(crowd_pressure / max(1e-6, cfg.pressure_critical))
        disorder_norm = self._clip01(disorder / max(0.001, cfg.disorder_soft_limit))
        bottleneck_norm = self._clip01(bottleneck / max(0.001, cfg.bottleneck_soft_limit))
        oscillation_norm = self._clip01(oscillation / max(0.001, cfg.oscillation_soft_limit))
        # Speed contributes through *deviation* from a comfortable walking pace:
        # both a frozen crowd and a sprinting crowd are abnormal.
        speed_norm = self._clip01(abs(avg_speed_ms - 1.2) / 1.2) if count > 0 else 0.0

        w = cfg.weights
        total_w = sum(w.as_dict().values()) or 1.0
        contributions = {
            "local_density": w.local_density * density_norm / total_w,
            "crowd_pressure": w.crowd_pressure * pressure_norm / total_w,
            "flow_disorder": w.flow_disorder * disorder_norm / total_w,
            "bottleneck": w.bottleneck * bottleneck_norm / total_w,
            "oscillation": w.oscillation * oscillation_norm / total_w,
            "speed": w.speed * speed_norm / total_w,
        }
        risk_score = self._clip01(sum(contributions.values()))

        # --- typology -----------------------------------------------------------
        # Below a handful of people every spatial statistic is dominated by
        # sampling noise, and claiming a failure mode from four data points
        # would be theatre rather than measurement.
        if count < 5:
            self._last_time = float(timestamp_sec)
            from .risk_taxonomy import RiskTypeScore
            normal = RiskTypeScore("NORMAL_FLOW", "Normal Flow", 1.0, 0.05, "\u2713", "#2ecc71",
                                   [f"only {count} people tracked -- below the analysis threshold"])
            typology = TypologyResult(primary=normal, ranked=[normal], hazard_index=0.0)
            risk_score = self._clip01(min(risk_score, 0.2))
            level = "low"
            return SceneFeatures(
                frame_id=frame_id, timestamp_sec=float(timestamp_sec), person_count=count,
                density=float(density_global), local_density_peak=float(local_density_peak),
                local_density_mean=float(local_density_mean), avg_speed=float(avg_speed_norm),
                avg_speed_ms=float(avg_speed_ms), flow_disorder=float(disorder),
                counterflow_index=0.0, bottleneck_concentration=1.0, lateral_spread=1.0,
                bottleneck_ratio=float(bottleneck), crowd_pressure=float(crowd_pressure),
                flux=float(flux), flux_efficiency=float(flux_efficiency),
                oscillation_index=float(oscillation), density_rate_per_min=float(density_rate),
                count_rate_per_min=float(count_rate), speed_surge_ratio=float(surge),
                sustained_density_sec=float(self._sustained_density_sec),
                trend_summary=self.describe_change(self.change_since(60.0)),
                risk_score=risk_score, risk_level=level,
                factors=["insufficient crowd for meaningful risk analysis"],
                primary_risk_type="NORMAL_FLOW", primary_risk_label="Normal Flow",
                primary_risk_score=1.0, hazard_index=0.0,
                risk_type_ranking=[normal.to_dict()], type_evidence=normal.evidence,
                zones=[z.to_dict() for z in zone_readings], hotspot_zone="", hotspot_density=0.0,
                contributions={k: round(v, 4) for k, v in contributions.items()},
                calibrated=ground.calibrated, calibration_note=ground.describe(),
            )

        typology: TypologyResult = self.classifier.classify(
            {
                "local_density_peak": local_density_peak,
                "density": local_density_mean,
                "avg_speed_ms": avg_speed_ms,
                "flow_disorder": disorder,
                "bottleneck_ratio": bottleneck,
                "crowd_pressure": crowd_pressure,
                "oscillation_index": oscillation,
                "flux_efficiency": flux_efficiency,
                "density_rate_per_min": density_rate,
                "count_rate_per_min": count_rate,
                "speed_surge_ratio": surge,
                "counterflow_index": counterflow_index,
                "bottleneck_concentration": bottleneck_concentration,
                "lateral_spread": lateral_spread,
                "sustained_density_sec": self._sustained_density_sec,
            }
        )

        # The hazard index can lift the score when several failure modes
        # co-occur -- a crowd that is dense AND turbulent AND jammed is worse
        # than the linear sum of those features suggests.
        risk_score = self._clip01(max(risk_score, 0.55 * risk_score + 0.45 * typology.hazard_index))

        if risk_score >= cfg.high_threshold:
            level = "high"
        elif risk_score >= cfg.low_threshold:
            level = "moderate"
        else:
            level = "low"

        # --- per-zone scoring ------------------------------------------------------
        for z in zone_readings:
            z_density_norm = self._clip01((z.density_pm2 - 0.5) / max(0.1, cfg.density_hard_limit - 0.5))
            z_press_norm = self._clip01(z.pressure / max(1e-6, cfg.pressure_critical))
            z_dis_norm = self._clip01(z.flow_disorder / max(0.001, cfg.disorder_soft_limit))
            z_load = self._clip01(z.load_factor)
            z.risk_score = self._clip01(
                0.40 * z_density_norm + 0.25 * z_press_norm + 0.20 * z_dis_norm + 0.15 * z_load
            )
            z.risk_level = (
                "high" if z.risk_score >= cfg.high_threshold
                else "moderate" if z.risk_score >= cfg.low_threshold
                else "low"
            )
        hotspot = self.zone_manager.hottest(zone_readings) if self.zone_manager else None

        # --- human-readable factors ---------------------------------------------
        factors: List[str] = []
        if local_density_peak >= cfg.density_hard_limit:
            factors.append(f"peak local density {local_density_peak:.1f}/m2 above the safe operating limit")
        elif local_density_peak >= cfg.density_soft_limit:
            factors.append(f"high local crowd density {local_density_peak:.1f}/m2")
        if crowd_pressure >= cfg.pressure_critical:
            factors.append(f"crowd pressure {crowd_pressure:.4f} above the turbulence threshold")
        elif crowd_pressure >= cfg.pressure_warning:
            factors.append(f"crowd pressure {crowd_pressure:.4f} rising toward the turbulence threshold")
        if disorder >= cfg.disorder_soft_limit:
            factors.append(f"counter-flow / directional disorder {disorder:.2f}")
        if bottleneck >= cfg.bottleneck_soft_limit:
            factors.append(f"{bottleneck * 100:.0f}% of the crowd concentrated in a constriction")
        if oscillation >= cfg.oscillation_soft_limit:
            factors.append(f"stop-and-go wave activity {oscillation:.2f}")
        if surge >= cfg.speed_surge_ratio:
            factors.append(f"movement speed surged to {surge:.1f}x baseline")
        if density_rate >= 0.5:
            factors.append(f"density climbing {density_rate:.2f}/m2 per minute")
        if not factors:
            factors.append("normal movement and density pattern")

        return SceneFeatures(
            frame_id=frame_id,
            timestamp_sec=float(timestamp_sec),
            person_count=count,
            density=float(density_global),
            local_density_peak=float(local_density_peak),
            local_density_mean=float(local_density_mean),
            avg_speed=float(avg_speed_norm),
            avg_speed_ms=float(avg_speed_ms),
            flow_disorder=float(disorder),
            counterflow_index=float(counterflow_index),
            bottleneck_ratio=float(bottleneck),
            bottleneck_concentration=float(bottleneck_concentration),
            lateral_spread=float(lateral_spread),
            crowd_pressure=float(crowd_pressure),
            flux=float(flux),
            flux_efficiency=float(flux_efficiency),
            oscillation_index=float(oscillation),
            density_rate_per_min=float(density_rate),
            count_rate_per_min=float(count_rate),
            speed_surge_ratio=float(surge),
            sustained_density_sec=float(self._sustained_density_sec),
            trend_summary=self.describe_change(self.change_since(60.0)),
            hotspot_alternate=hotspot.alternate if hotspot else "",
            hotspot_meter_at=hotspot.meter_at if hotspot else "",
            hotspot_staging=hotspot.staging if hotspot else "",
            risk_score=float(risk_score),
            risk_level=level,
            factors=factors,
            primary_risk_type=typology.primary.code,
            primary_risk_label=typology.primary.label,
            primary_risk_score=float(typology.primary.score),
            hazard_index=float(typology.hazard_index),
            risk_type_ranking=[t.to_dict() for t in typology.ranked],
            type_evidence=typology.primary.evidence,
            zones=[z.to_dict() for z in zone_readings],
            hotspot_zone=hotspot.name if hotspot else "",
            hotspot_density=float(hotspot.density_pm2) if hotspot else 0.0,
            contributions={k: round(v, 4) for k, v in contributions.items()},
            calibrated=ground.calibrated,
            calibration_note=ground.describe(),
        )

    # ------------------------------------------------------------------ #
    def change_since(self, seconds: float = 60.0) -> Optional[Dict[str, Any]]:
        """What has changed over the last `seconds`, in operator language.

        This is the measurement a person watching the same camera cannot make.
        They see "a lot of people"; this says density went from 2.8 to 4.1 in
        sixty seconds while speed fell 35 per cent. The first is an impression,
        the second is a trend somebody can act on.
        """
        if len(self._long_hist) < 4:
            return None
        now_t, now_d, now_v, now_n = self._long_hist[-1]
        target = now_t - seconds

        past = None
        for entry in self._long_hist:
            if entry[0] >= target:
                past = entry
                break
        if past is None or (now_t - past[0]) < seconds * 0.4:
            past = self._long_hist[0]
        elapsed = now_t - past[0]
        if elapsed < 1.0:
            return None

        then_t, then_d, then_v, then_n = past

        def pct(new: float, old: float) -> Optional[float]:
            return ((new - old) / old * 100.0) if old > 1e-6 else None

        return {
            "window_sec": round(elapsed, 1),
            "density_then": round(then_d, 2), "density_now": round(now_d, 2),
            "density_change_pct": pct(now_d, then_d),
            "speed_then": round(then_v, 2), "speed_now": round(now_v, 2),
            "speed_change_pct": pct(now_v, then_v),
            "count_then": then_n, "count_now": now_n,
            "count_change": now_n - then_n,
        }

    @staticmethod
    def describe_change(change: Optional[Dict[str, Any]]) -> str:
        if not change:
            return ""
        bits = [f"Over the last {change['window_sec']:.0f}s: density "
                f"{change['density_then']:.1f} to {change['density_now']:.1f} persons/m2"]
        if change["density_change_pct"] is not None:
            bits[-1] += f" ({change['density_change_pct']:+.0f}%)"
        bits.append(f"mean speed {change['speed_then']:.2f} to {change['speed_now']:.2f} m/s")
        if change["speed_change_pct"] is not None:
            bits[-1] += f" ({change['speed_change_pct']:+.0f}%)"
        bits.append(f"occupancy {change['count_then']} to {change['count_now']} "
                    f"({change['count_change']:+d})")
        return "; ".join(bits) + "."

    def scene_query(self, features: SceneFeatures, typology_terms: str = "") -> str:
        """Query string handed to the RAG retriever.

        Includes the failure-mode vocabulary so retrieval pulls the SOP for the
        specific hazard rather than generic crowding advice.
        """
        zone_part = f"zone {features.hotspot_zone}; " if features.hotspot_zone else ""
        return (
            f"{features.primary_risk_label}; crowd risk {features.risk_level}; {zone_part}"
            f"count {features.person_count}; "
            f"peak local density {features.local_density_peak:.2f} persons per square meter; "
            f"speed {features.avg_speed_ms:.2f} m/s; flow disorder {features.flow_disorder:.2f}; "
            f"bottleneck {features.bottleneck_ratio:.2f}; crowd pressure {features.crowd_pressure:.3f}; "
            f"counter-flow index {features.counterflow_index:.2f}; "
            f"oscillation {features.oscillation_index:.2f}; "
            f"factors: {', '.join(features.factors)}. {typology_terms}"
        ).strip()
