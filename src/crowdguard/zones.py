"""Named ground zones.

An alert that says "the video is risky" is not actionable. An alert that says
"Gate B, 4.6 persons/m2, counter-flow, throughput collapsing" tells a marshal
where to walk. This module replaces the old hard-coded "central 40% x 65% of
the image" bottleneck proxy with operator-defined polygons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ZoneConfig


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray-casting test. Pure numpy, no cv2 dependency."""
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    n = polygon.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        straddles = (yi > y) != (yj > y)
        denom = np.where(np.abs(yj - yi) < 1e-12, 1e-12, yj - yi)
        x_cross = (xj - xi) * (y - yi) / denom + xi
        inside ^= straddles & (x < x_cross)
        j = i
    return inside


@dataclass
class Zone:
    """A named area, plus where to send people when it fails.

    Routing is part of the zone definition rather than something the advisor
    invents, because only the venue knows its own topology. Without it an
    advisory can say "Gate 3 is congested" but not "divert to Gate 4" -- and
    the second is the sentence a marshal can act on. These three fields are the
    minimum needed to turn a diagnosis into a dispatch:

        alternate  where to divert people to
        meter_at   the upstream point at which to hold arriving flow
        staging    where marshals should assemble to work this zone
    """

    name: str
    kind: str = "area"                      # gate | corridor | exit | stage | area
    polygon: List[List[float]] = field(default_factory=list)   # normalised 0-1
    capacity_pm2: float = 4.0
    alternate: str = ""
    meter_at: str = ""
    staging: str = ""

    def pixel_polygon(self, frame_shape: Tuple[int, ...]) -> np.ndarray:
        h, w = float(frame_shape[0]), float(frame_shape[1])
        return np.asarray([[p[0] * w, p[1] * h] for p in self.polygon], dtype=np.float64)

    def normalised_area(self) -> float:
        """Shoelace area in normalised units (fraction of the frame)."""
        p = np.asarray(self.polygon, dtype=np.float64)
        if p.shape[0] < 3:
            return 0.0
        x, y = p[:, 0], p[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


@dataclass
class ZoneReading:
    """Per-zone measurement for one frame."""

    name: str
    kind: str
    count: int
    occupancy_ratio: float          # share of all detected people in this zone
    area_fraction: float            # share of the frame this zone covers
    concentration: float            # occupancy share / area share; 1.0 == uniform
    density_pm2: float              # peak local density inside the zone
    mean_speed_ms: float
    flow_disorder: float
    pressure: float
    load_factor: float              # density / capacity, 1.0 == at design limit
    risk_score: float = 0.0
    risk_level: str = "low"
    alternate: str = ""
    meter_at: str = ""
    staging: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "count": self.count,
            "occupancy_ratio": round(self.occupancy_ratio, 4),
            "area_fraction": round(self.area_fraction, 4),
            "concentration": round(self.concentration, 3),
            "density_pm2": round(self.density_pm2, 3),
            "mean_speed_ms": round(self.mean_speed_ms, 3),
            "flow_disorder": round(self.flow_disorder, 3),
            "pressure": round(self.pressure, 5),
            "load_factor": round(self.load_factor, 3),
            "risk_score": round(self.risk_score, 3),
            "risk_level": self.risk_level,
            "alternate": self.alternate,
            "meter_at": self.meter_at,
            "staging": self.staging,
        }


class ZoneManager:
    """Holds the zone layout and computes per-zone readings."""

    def __init__(self, config: Optional[ZoneConfig] = None):
        config = config or ZoneConfig()
        layout = config.zones or ZoneConfig.default_layout()
        self.enabled = config.enabled
        self.zones: List[Zone] = [
            Zone(
                name=z.get("name", f"Zone {i + 1}"),
                kind=z.get("kind", "area"),
                polygon=[list(map(float, p)) for p in z.get("polygon", [])],
                capacity_pm2=float(z.get("capacity_pm2", 4.0)),
                alternate=str(z.get("alternate", "")),
                meter_at=str(z.get("meter_at", "")),
                staging=str(z.get("staging", "")),
            )
            for i, z in enumerate(layout)
        ]
        self.zones = [z for z in self.zones if len(z.polygon) >= 3]

    def __len__(self) -> int:
        return len(self.zones)

    def assign(self, centroids: Sequence[Tuple[float, float]], frame_shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """Return {zone_name: boolean mask over centroids}."""
        pts = np.asarray(list(centroids), dtype=np.float64).reshape(-1, 2)
        return {z.name: points_in_polygon(pts, z.pixel_polygon(frame_shape)) for z in self.zones}

    def readings(
        self,
        centroids: Sequence[Tuple[float, float]],
        velocities_ms: np.ndarray,
        per_person_density: np.ndarray,
        per_person_pressure: np.ndarray,
        frame_shape: Tuple[int, ...],
        percentile: float = 90.0,
    ) -> List[ZoneReading]:
        pts = np.asarray(list(centroids), dtype=np.float64).reshape(-1, 2)
        total = max(1, pts.shape[0])
        masks = self.assign(pts, frame_shape)
        out: List[ZoneReading] = []

        for zone in self.zones:
            mask = masks[zone.name]
            n = int(mask.sum())
            if n == 0:
                out.append(
                    ZoneReading(zone.name, zone.kind, 0, 0.0, zone.normalised_area(), 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "low",
                                zone.alternate, zone.meter_at, zone.staging)
                )
                continue

            dens = float(np.percentile(per_person_density[mask], percentile)) if per_person_density.size else 0.0
            press = float(np.percentile(per_person_pressure[mask], percentile)) if per_person_pressure.size else 0.0

            if velocities_ms.size:
                v = velocities_ms[mask]
                speeds = np.linalg.norm(v, axis=1)
                mean_speed = float(speeds.mean())
                moving = speeds > 1e-4
                if moving.sum() > 1:
                    resultant = float(np.linalg.norm(v[moving].sum(axis=0)))
                    disorder = float(np.clip(1.0 - resultant / (speeds[moving].sum() + 1e-9), 0.0, 1.0))
                else:
                    disorder = 0.0
            else:
                mean_speed, disorder = 0.0, 0.0

            area_fraction = max(1e-3, zone.normalised_area())
            occupancy = n / total
            out.append(
                ZoneReading(
                    name=zone.name,
                    kind=zone.kind,
                    count=n,
                    occupancy_ratio=occupancy,
                    area_fraction=area_fraction,
                    # Concentration, not occupancy share. A zone holding 40% of
                    # the crowd is only a hotspot if it is smaller than 40% of
                    # the area -- otherwise it is simply a big zone.
                    concentration=occupancy / area_fraction,
                    density_pm2=dens,
                    mean_speed_ms=mean_speed,
                    flow_disorder=disorder,
                    pressure=press,
                    load_factor=dens / max(0.1, zone.capacity_pm2),
                    alternate=zone.alternate,
                    meter_at=zone.meter_at,
                    staging=zone.staging,
                )
            )
        return out

    def hottest(self, readings: Sequence[ZoneReading]) -> Optional[ZoneReading]:
        if not readings:
            return None
        return max(readings, key=lambda r: r.risk_score)

    def as_layout(self) -> List[Dict[str, Any]]:
        return [
            {"name": z.name, "kind": z.kind, "polygon": z.polygon,
             "capacity_pm2": z.capacity_pm2, "alternate": z.alternate,
             "meter_at": z.meter_at, "staging": z.staging}
            for z in self.zones
        ]
