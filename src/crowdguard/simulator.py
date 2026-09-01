"""Agent-based crowd simulator.

Why a simulator is part of the deliverable, not a toy
-----------------------------------------------------
Three problems it solves that nothing else can:

1. **Demonstrability.** Real crowd-disaster footage is scarce, ethically
   fraught and impossible to obtain on demand. Every failure mode the system
   claims to detect can be produced here on command, which is the difference
   between claiming the classifier works and showing it work.

2. **Ground truth.** The simulator knows the true count and the true local
   density of every agent, because it placed them. That makes it possible to
   report a real counting error and a real density error -- measured numbers,
   not assertions -- for the parts of the system that sit downstream of
   detection.

3. **Falsifiability.** A rule-based classifier is easy to fool yourself about.
   Scripted scenarios with known correct answers make the classifier testable,
   and `scripts/evaluate.py` uses exactly these scenarios as its test set.

Physics: a reduced social-force model (Helbing & Molnar, 1995). Agents are
driven toward a goal, repelled from each other with an exponential kernel, and
repelled from walls. It is not a research-grade pedestrian simulator, and it is
not presented as one -- it reproduces the qualitative regimes (free flow,
congested branch, jamming, counter-flow interlocking) well enough to exercise
the detector-free part of the pipeline.

Everything produced here is labelled `simulated=True` all the way to the UI.
It is never presented as camera output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# World is a 12 m wide by 20 m deep ground plane. Gate aperture sits at y = 7 m.
WORLD_W = 12.0
WORLD_D = 20.0

# Synthetic camera: a trapezoid on the ground maps to the full image, giving
# realistic perspective foreshortening. These four correspondences double as
# the calibration file shipped for the simulated feed.
FRAME_W, FRAME_H = 960, 540


def camera_correspondences() -> Dict[str, List[List[float]]]:
    """Image <-> world point pairs describing the synthetic camera."""
    return {
        "image_points": [[80.0, 520.0], [880.0, 520.0], [610.0, 150.0], [350.0, 150.0]],
        "world_points": [[0.0, 0.0], [WORLD_W, 0.0], [WORLD_W, WORLD_D], [0.0, WORLD_D]],
    }


def _world_to_image_matrix() -> np.ndarray:
    from .calibration import _solve_homography

    c = camera_correspondences()
    # Inverse direction: world -> image.
    return _solve_homography(np.asarray(c["world_points"]), np.asarray(c["image_points"]))


@dataclass
class Scenario:
    key: str
    label: str
    description: str
    expected_type: str
    duration_sec: float = 60.0


SCENARIOS: Dict[str, Scenario] = {
    "normal_flow": Scenario(
        "normal_flow", "Normal Flow",
        "A steady, comfortable stream walking one way through an open corridor.",
        "NORMAL_FLOW", 45.0),
    "rapid_influx": Scenario(
        "rapid_influx", "Rapid Influx",
        "Occupancy climbing quickly from an empty area. Still safe, but the trend is not.",
        "RAPID_INFLUX", 50.0),
    "bottleneck": Scenario(
        "bottleneck", "Bottleneck Congestion",
        "A 1.6 m gate cannot discharge people as fast as they arrive; a queue forms and flux collapses.",
        "BOTTLENECK_CONGESTION", 70.0),
    "counterflow": Scenario(
        "counterflow", "Counter-Flow Conflict",
        "Two opposing streams meet in one corridor and interlock.",
        "COUNTERFLOW_CONFLICT", 60.0),
    "static_blockage": Scenario(
        "static_blockage", "Static Blockage",
        "The route ahead closes. A dense crowd stops moving while people keep arriving behind it.",
        "STATIC_BLOCKAGE", 60.0),
    "turbulent_surge": Scenario(
        "turbulent_surge", "Turbulent Surge",
        "A dense crowd with stop-and-go shock waves propagating back through it.",
        "TURBULENT_SURGE", 70.0),
    "panic_dispersal": Scenario(
        "panic_dispersal", "Panic Dispersal",
        "A trigger appears and the crowd scatters away from it at speed.",
        "PANIC_DISPERSAL", 45.0),
    "escalating_crush": Scenario(
        "escalating_crush", "Escalating Crush (full arc)",
        "The complete narrative: normal flow, rising influx, a gate bottleneck, jamming, "
        "and finally a progressive crush. This is the scenario to use for a demo.",
        "PROGRESSIVE_CRUSH", 120.0),
}


@dataclass
class SimFrame:
    """One simulated observation, shaped exactly like a detector's output."""

    frame_id: int
    timestamp_sec: float
    frame: np.ndarray
    detections: List[Dict[str, Any]]
    true_count: int
    true_density_peak: float
    true_density_mean: float
    phase: str
    simulated: bool = True

    def ground_truth(self) -> Dict[str, Any]:
        return {
            "true_count": self.true_count,
            "true_density_peak": round(self.true_density_peak, 3),
            "true_density_mean": round(self.true_density_mean, 3),
            "phase": self.phase,
        }


class CrowdSimulator:
    """Social-force crowd with scripted scenarios."""

    def __init__(
        self,
        scenario: str = "escalating_crush",
        fps: float = 10.0,
        seed: int = 7,
        detection_noise_px: float = 3.0,
        occlusion_dropout: bool = True,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Options: {sorted(SCENARIOS)}")
        self.scenario = SCENARIOS[scenario]
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.rng = np.random.default_rng(seed)
        self.detection_noise_px = detection_noise_px
        self.occlusion_dropout = occlusion_dropout

        self.H = _world_to_image_matrix()
        self.frame_id = -1
        self.t = 0.0

        self.pos = np.zeros((0, 2), dtype=np.float64)   # metres, x across / y depth
        self.vel = np.zeros((0, 2), dtype=np.float64)
        self.pref_speed = np.zeros((0,), dtype=np.float64)
        self.direction = np.zeros((0,), dtype=np.float64)  # +1 toward y=0, -1 away
        self.background = self._make_background()

        self._gate_open = True
        self._panic_origin: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    @property
    def total_frames(self) -> int:
        return int(self.scenario.duration_sec * self.fps)

    def _make_background(self) -> np.ndarray:
        import cv2

        bg = np.full((FRAME_H, FRAME_W, 3), 58, dtype=np.uint8)
        # Ground plane gradient, so perspective reads correctly.
        for y in range(FRAME_H):
            shade = int(40 + 55 * (y / FRAME_H))
            bg[y, :, :] = (shade, shade - 4, shade - 8)

        # Paving grid drawn in world coordinates, every 2 m.
        for gx in np.arange(0, WORLD_W + 0.1, 2.0):
            pts = self.world_to_image(np.array([[gx, 0.0], [gx, WORLD_D]]))
            cv2.line(bg, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), (78, 74, 70), 1, cv2.LINE_AA)
        for gy in np.arange(0, WORLD_D + 0.1, 2.0):
            pts = self.world_to_image(np.array([[0.0, gy], [WORLD_W, gy]]))
            cv2.line(bg, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), (78, 74, 70), 1, cv2.LINE_AA)
        return bg

    # ------------------------------------------------------------------ #
    def world_to_image(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((0, 2))
        hom = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ self.H.T
        w = np.where(np.abs(hom[:, 2:3]) < 1e-9, 1e-9, hom[:, 2:3])
        return hom[:, :2] / w

    def _pixel_height(self, world_pt: np.ndarray) -> float:
        """Apparent height in pixels of a 1.7 m person standing at this point."""
        foot = self.world_to_image(world_pt.reshape(1, 2))[0]
        # Approximate: project a point 1.7 m "further away" along depth is wrong;
        # instead scale by the local vertical gradient of the homography.
        near = self.world_to_image(np.array([[world_pt[0], world_pt[1]]]))[0]
        far = self.world_to_image(np.array([[world_pt[0], min(WORLD_D, world_pt[1] + 1.0)]]))[0]
        depth_scale = abs(near[1] - far[1])
        return float(np.clip(depth_scale * 1.7, 14.0, 260.0)) if depth_scale > 0 else 40.0

    # ------------------------------------------------------------------ #
    def _spawn(self, n: int, band: Tuple[float, float], direction: float, speed: float = 1.25) -> None:
        if n <= 0:
            return
        x = self.rng.uniform(0.6, WORLD_W - 0.6, n)
        y = self.rng.uniform(band[0], band[1], n)
        self.pos = np.vstack([self.pos, np.column_stack([x, y])])
        self.vel = np.vstack([self.vel, np.zeros((n, 2))])
        self.pref_speed = np.concatenate([self.pref_speed, self.rng.normal(speed, 0.15, n).clip(0.4, 2.2)])
        self.direction = np.concatenate([self.direction, np.full(n, direction)])

    def _despawn(self, mask: np.ndarray) -> None:
        keep = ~mask
        self.pos, self.vel = self.pos[keep], self.vel[keep]
        self.pref_speed, self.direction = self.pref_speed[keep], self.direction[keep]

    # ------------------------------------------------------------------ #
    def _script(self) -> Tuple[str, Dict[str, Any]]:
        """Scenario schedule: returns (phase name, control parameters)."""
        s, t = self.scenario.key, self.t
        n = self.pos.shape[0]

        if s == "normal_flow":
            return "steady", {"spawn": 1 if n < 34 else 0, "gate": None, "speed": 1.3}

        if s == "rapid_influx":
            # Fills continuously for the whole run: the point of this scenario
            # is the trend, so it must not plateau half way through.
            return "filling", {"spawn": 3 if n < 260 else 0, "gate": None, "speed": 1.25}

        if s == "bottleneck":
            return "queueing", {"spawn": 2 if n < 58 else 0, "gate": 2.4, "speed": 1.3}

        if s == "counterflow":
            return "interlocked", {"spawn": 0, "gate": None, "speed": 1.2, "counter": True,
                                   "spawn_both": 1 if n < 90 else 0}

        if s == "static_blockage":
            if t < 15:
                return "flowing", {"spawn": 2 if n < 55 else 1, "gate": None, "speed": 1.3}
            return "jammed", {"spawn": 2 if n < 95 else 0, "gate": None, "stop_line": 6.0, "speed": 1.3}

        if s == "turbulent_surge":
            if t < 12:
                return "filling", {"spawn": 3 if n < 70 else 0, "gate": 2.6, "speed": 1.35}
            # Alternate the gate open/shut to inject stop-and-go shock waves.
            open_now = (int(t) // 3) % 2 == 0
            # Alternating aperture drives stop-and-go shock waves back through
            # the crowd -- the mechanism of crowd turbulence.
            return "surging", {"spawn": 2 if n < 78 else 0, "gate": 4.0 if open_now else 0.5,
                               "speed": 1.6}

        if s == "panic_dispersal":
            if t < 15:
                return "calm", {"spawn": 3 if n < 65 else 0, "gate": None, "speed": 1.1}
            if self._panic_origin is None:
                self._panic_origin = np.array([WORLD_W / 2, WORLD_D * 0.35])
            return "fleeing", {"spawn": 3 if n < 70 else 1, "gate": None, "speed": 2.6, "panic": True}

        # escalating_crush -- the full narrative arc
        if t < 20:
            return "normal", {"spawn": 1 if n < 35 else 0, "gate": None, "speed": 1.35}
        if t < 45:
            return "influx", {"spawn": 4 if n < 110 else 1, "gate": 3.0, "speed": 1.3}
        if t < 75:
            return "bottleneck", {"spawn": 3 if n < 120 else 1, "gate": 1.5, "speed": 1.25}
        if t < 100:
            return "jamming", {"spawn": 2 if n < 150 else 0, "gate": 0.7, "speed": 1.2}
        return "crush", {"spawn": 1 if n < 170 else 0, "gate": 0.35, "speed": 1.1}

    # ------------------------------------------------------------------ #
    GATE_Y = 7.0
    FUNNEL_LEN = 5.5
    THROAT_LEN = 1.8

    def _wall_half_width(self, y: np.ndarray, gate: Optional[float]) -> np.ndarray:
        """Half-width of the walkable corridor at each depth y.

        With a gate configured the corridor converges linearly into the
        aperture and then widens again downstream. Compression happens here,
        for the same reason it happens at a real gate: the walls do the work,
        not a scripted "everyone slow down" rule.
        """
        half_open = np.full_like(y, WORLD_W / 2.0 - 0.3)
        if gate is None:
            return half_open

        g = max(0.25, float(gate)) / 2.0
        gy, fl, tl = self.GATE_Y, self.FUNNEL_LEN, self.THROAT_LEN

        out = half_open.copy()

        # Converging funnel upstream of the gate.
        conv = (y >= gy) & (y < gy + fl)
        frac = np.clip((y[conv] - gy) / fl, 0.0, 1.0)
        out[conv] = g + (half_open[conv] - g) * frac

        # Throat.
        throat = (y >= gy - tl) & (y < gy)
        out[throat] = g

        # Widening discharge downstream.
        disc = y < gy - tl
        frac_d = np.clip((gy - tl - y[disc]) / 3.0, 0.0, 1.0)
        out[disc] = g + (half_open[disc] - g) * frac_d
        return out

    def _step_physics(self, ctrl: Dict[str, Any]) -> None:
        n = self.pos.shape[0]
        if n == 0:
            return

        goal_speed = float(ctrl.get("speed", 1.3))
        gate = ctrl.get("gate")
        stop_line = ctrl.get("stop_line")
        centre_x = WORLD_W / 2.0
        desired = np.zeros_like(self.pos)

        if ctrl.get("panic") and self._panic_origin is not None:
            away = self.pos - self._panic_origin
            norm = np.linalg.norm(away, axis=1, keepdims=True)
            desired = away / np.maximum(norm, 1e-6) * goal_speed
        else:
            desired[:, 1] = -self.direction * goal_speed
            if gate is not None:
                # Steer toward the aperture; the strength grows as the gate nears.
                ahead = self.pos[:, 1] - self.GATE_Y
                urgency = np.clip(1.0 - ahead / (self.FUNNEL_LEN + 3.0), 0.0, 1.0)
                dx = centre_x - self.pos[:, 0]
                desired[:, 0] += np.sign(dx) * np.minimum(np.abs(dx), 1.0) * urgency * goal_speed * 0.9

        # --- social repulsion (Helbing & Molnar exponential kernel) ----------
        # Deliberately weaker and shorter-ranged than the driving force: in a
        # real crush, pressure from behind overwhelms personal space, and a
        # model whose repulsion always wins can never reproduce compression.
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        close = dist < 1.10
        if close.any():
            unit = diff / np.maximum(dist, 1e-6)[..., None]
            magnitude = np.where(close, 0.95 * np.exp((0.36 - dist) / 0.20), 0.0)
            magnitude = np.clip(magnitude, 0.0, 3.5)
            desired += np.einsum("ij,ijk->ik", magnitude, unit) * 0.42

        # --- a full-width closure: the route ahead is simply shut ---------------
        # Distinct from a narrow gate. A gate funnels and compresses; a closure
        # stops a broad front of people where they stand, which is the physical
        # situation the STATIC_BLOCKAGE failure mode describes.
        if stop_line is not None:
            blocked = self.pos[:, 1] <= float(stop_line) + 1.2
            desired[blocked, 1] = 0.0

        # --- jamming: speed collapses as density rises -------------------------
        # The fundamental diagram of pedestrian flow (Weidmann; Fruin): walking
        # speed falls roughly linearly with density and reaches zero at the jam
        # density. Without this a simulated crowd walks at 1.3 m/s through a
        # 9 persons/m2 pack, which is physically impossible and would make the
        # crush scenario look like a surge.
        if n > 3:
            k = min(4, n - 1)
            dd = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
            np.fill_diagonal(dd, np.inf)
            dd.sort(axis=1)
            rho_local = k / (np.pi * np.maximum(dd[:, k - 1], 0.30) ** 2)
            jam = np.clip(1.0 - (rho_local - 3.5) / 5.0, 0.04, 1.0)
            desired *= jam[:, None]

        # --- corridor walls ---------------------------------------------------
        half = self._wall_half_width(self.pos[:, 1], gate)
        left, right = centre_x - half, centre_x + half
        desired[:, 0] += 3.2 * np.exp((0.30 - (self.pos[:, 0] - left)) / 0.22)
        desired[:, 0] -= 3.2 * np.exp((0.30 - (right - self.pos[:, 0])) / 0.22)

        speed = np.linalg.norm(desired, axis=1, keepdims=True)
        cap = 3.2 if ctrl.get("panic") else 2.2
        desired = np.where(speed > cap, desired / np.maximum(speed, 1e-6) * cap, desired)

        self.vel += (desired - self.vel) * min(1.0, self.dt / 0.40)
        self.pos += self.vel * self.dt

        # Hard wall constraint. Without this the crowd leaks through the funnel
        # and density never builds.
        half = self._wall_half_width(self.pos[:, 1], gate)
        left, right = centre_x - half, centre_x + half
        self.pos[:, 0] = np.clip(self.pos[:, 0], left + 0.12, right - 0.12)
        self.pos[:, 1] = np.clip(self.pos[:, 1], 0.25, WORLD_D - 0.25)

        # Bodies are not points. Enforcing a minimum centre-to-centre spacing
        # caps the achievable density at roughly 9-10 persons/m2, which is the
        # physical limit observed in real crush incidents; without it the model
        # happily produces impossible densities and the demo becomes a lie.
        self._enforce_separation()

        if ctrl.get("panic"):
            # Fleeing people leave through whichever edge they reach.
            edge = (
                (self.pos[:, 0] <= 0.45) | (self.pos[:, 0] >= WORLD_W - 0.45)
                | (self.pos[:, 1] <= 0.45) | (self.pos[:, 1] >= WORLD_D - 0.45)
            )
            if edge.any():
                self._despawn(edge)
        else:
            exited = (self.pos[:, 1] <= 0.4) & (self.direction > 0)
            exited |= (self.pos[:, 1] >= WORLD_D - 0.4) & (self.direction < 0)
            if exited.any():
                self._despawn(exited)

    def _enforce_separation(self, min_d: float = 0.42, iterations: int = 3) -> None:
        n = self.pos.shape[0]
        if n < 2:
            return
        for _ in range(iterations):
            diff = self.pos[:, None, :] - self.pos[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            overlapping = dist < min_d
            if not overlapping.any():
                return
            unit = diff / np.maximum(dist, 1e-6)[..., None]
            push = np.where(overlapping, (min_d - dist) * 0.5, 0.0)
            self.pos += np.einsum("ij,ijk->ik", push, unit) * 0.25
            self.pos[:, 1] = np.clip(self.pos[:, 1], 0.25, WORLD_D - 0.25)

    # ------------------------------------------------------------------ #
    def _true_densities(self) -> Tuple[float, float]:
        n = self.pos.shape[0]
        if n < 2:
            return 0.0, 0.0
        k = min(4, n - 1)
        d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        d.sort(axis=1)
        r = np.maximum(d[:, k - 1], 0.30)
        dens = np.clip(k / (np.pi * r ** 2), 0, 9.0)
        return float(np.percentile(dens, 90)), float(dens.mean())

    def _render(self, phase: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        import cv2

        frame = self.background.copy()
        detections: List[Dict[str, Any]] = []
        if self.pos.shape[0] == 0:
            return frame, detections

        img = self.world_to_image(self.pos)
        order = np.argsort(self.pos[:, 1])[::-1]      # far agents drawn first

        boxes: List[Tuple[float, float, float, float]] = []
        for idx in order:
            fx, fy = img[idx]
            if not (np.isfinite(fx) and np.isfinite(fy)):
                continue
            h_px = self._pixel_height(self.pos[idx])
            w_px = h_px * 0.34
            x1, y1 = fx - w_px / 2, fy - h_px
            x2, y2 = fx + w_px / 2, fy

            speed = float(np.linalg.norm(self.vel[idx]))
            body = (150, 120, 95) if self.direction[idx] > 0 else (95, 120, 155)
            if speed > 1.9:
                body = (110, 110, 205)
            cv2.rectangle(frame, (int(x1), int(y1 + h_px * 0.28)), (int(x2), int(y2)), body, -1)
            cv2.circle(frame, (int(fx), int(y1 + h_px * 0.16)), max(2, int(h_px * 0.15)),
                       (168, 148, 128), -1, cv2.LINE_AA)
            boxes.append((x1, y1, x2, y2))

        # Detector emulation: jitter, and drop heavily occluded agents so the
        # simulated feed reproduces the undercount a real detector suffers.
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if self.occlusion_dropout and i < len(boxes) - 1:
                overlap = 0.0
                for (ox1, oy1, ox2, oy2) in boxes[i + 1:]:
                    iw = max(0.0, min(x2, ox2) - max(x1, ox1))
                    ih = max(0.0, min(y2, oy2) - max(y1, oy1))
                    area = max(1e-6, (x2 - x1) * (y2 - y1))
                    overlap = max(overlap, iw * ih / area)
                if overlap > 0.72 and self.rng.random() < 0.55:
                    continue

            j = self.rng.normal(0.0, self.detection_noise_px, 4)
            detections.append(
                {
                    "bbox": [float(x1 + j[0]), float(y1 + j[1]), float(x2 + j[2]), float(y2 + j[3])],
                    "confidence": float(np.clip(self.rng.normal(0.82, 0.08), 0.35, 0.99)),
                    "class_id": 0,
                    "label": "person",
                }
            )

        cv2.putText(frame, f"SIMULATION - {self.scenario.label}  [{phase}]", (16, FRAME_H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (110, 210, 245), 1, cv2.LINE_AA)
        return frame, detections

    # ------------------------------------------------------------------ #
    def step(self) -> SimFrame:
        self.frame_id += 1
        self.t = self.frame_id / self.fps

        phase, ctrl = self._script()

        if ctrl.get("counter"):
            k = int(ctrl.get("spawn_both", 0))
            self._spawn(k, (WORLD_D - 3.0, WORLD_D - 0.6), +1.0, ctrl.get("speed", 1.2))
            self._spawn(k, (0.6, 3.0), -1.0, ctrl.get("speed", 1.2))
        else:
            self._spawn(int(ctrl.get("spawn", 0)), (WORLD_D - 3.0, WORLD_D - 0.6), +1.0, ctrl.get("speed", 1.3))

        self._step_physics(ctrl)
        peak, mean = self._true_densities()
        frame, detections = self._render(phase)

        return SimFrame(
            frame_id=self.frame_id,
            timestamp_sec=self.t,
            frame=frame,
            detections=detections,
            true_count=int(self.pos.shape[0]),
            true_density_peak=peak,
            true_density_mean=mean,
            phase=phase,
        )

    def run(self, frames: Optional[int] = None):
        total = frames if frames is not None else self.total_frames
        for _ in range(total):
            yield self.step()

    # ------------------------------------------------------------------ #
    @staticmethod
    def calibration_payload() -> Dict[str, Any]:
        """The exact calibration for the synthetic camera.

        Shipping this means the simulated feed can be run in fully calibrated
        mode, where reported persons/m2 can be checked against the simulator's
        own ground truth.
        """
        c = camera_correspondences()
        return {
            "note": "Ground-truth homography for the CrowdGuard simulator camera.",
            "image_points": c["image_points"],
            "world_points": c["world_points"],
            "world_extent_m": [WORLD_W, WORLD_D],
            "area_m2": WORLD_W * WORLD_D,
        }

    @staticmethod
    def zone_layout() -> List[Dict[str, Any]]:
        """Zones matching the simulator's geometry (the gate sits at y = 8 m)."""
        return [
            {"name": "Approach", "kind": "corridor",
             "polygon": [[0.16, 0.26], [0.86, 0.26], [0.94, 0.52], [0.07, 0.52]], "capacity_pm2": 4.0,
             "alternate": "Discharge", "meter_at": "Approach entry", "staging": "Approach edge"},
            {"name": "Gate Throat", "kind": "gate",
             "polygon": [[0.07, 0.52], [0.94, 0.52], [0.99, 0.74], [0.02, 0.74]], "capacity_pm2": 3.0,
             "alternate": "Discharge", "meter_at": "Approach", "staging": "Gate Throat downstream side"},
            {"name": "Discharge", "kind": "exit",
             "polygon": [[0.02, 0.74], [0.99, 0.74], [1.00, 1.00], [0.00, 1.00]], "capacity_pm2": 4.5,
             "alternate": "Gate Throat", "meter_at": "Gate Throat", "staging": "Discharge apron"},
        ]
