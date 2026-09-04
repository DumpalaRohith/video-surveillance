"""
events.py
Zone-based event detection: intrusion + loitering.
Point-in-polygon via shapely. Per-track state machine for loitering timers.
Also exposes live per-frame zone occupancy status for visualization.
"""

import json
from dataclasses import dataclass
from typing import List, Dict
from shapely.geometry import Point, Polygon


@dataclass
class Event:
    frame_number: int
    timestamp: float
    track_id: int
    bbox: tuple
    event_type: str      # "intrusion" or "loitering"
    zone_name: str
    confidence: float


@dataclass
class Zone:
    name: str
    zone_type: str
    polygon: Polygon
    loiter_threshold_seconds: float = None


@dataclass
class ZoneStatus:
    """Live status of a track relative to a zone, for visualization only —
    not written to the event log."""
    zone_name: str
    zone_type: str
    elapsed_seconds: float = 0.0
    threshold_seconds: float = None
    just_triggered: bool = False   # True for a short window right after an event fires


ALERT_FLASH_FRAMES = 20  # ~0.65s at 30fps — how long the "just entered" highlight lasts


class EventDetector:
    def __init__(self, zones_path: str, fps: float = 30.0):
        self.zones = self._load_zones(zones_path)
        self.fps = fps

        self._zone_entry_frame: Dict[tuple, int] = {}   # (track_id, zone_name) -> frame entered
        self._alerted: set = set()                       # (track_id, zone_name, event_type)
        self._alert_fired_frame: Dict[tuple, int] = {}   # same key -> frame it fired

    def _load_zones(self, zones_path: str) -> List[Zone]:
        with open(zones_path) as f:
            data = json.load(f)
        zones = []
        for z in data["zones"]:
            zones.append(Zone(
                name=z["name"],
                zone_type=z["type"],
                polygon=Polygon(z["polygon"]),
                loiter_threshold_seconds=z.get("loiter_threshold_seconds"),
            ))
        return zones

    def _bbox_center(self, bbox: tuple) -> Point:
        x1, y1, x2, y2 = bbox
        # bottom-center (feet position) — correct proxy for "which zone
        # is this person standing in", unlike full-box centroid
        return Point((x1 + x2) / 2, y2)

    def update(self, frame_number: int, tracks: list) -> List[Event]:
        """Call once per frame. Returns NEW events only (for the event log)."""
        events = []
        for track in tracks:
            point = self._bbox_center(track.bbox)
            for zone in self.zones:
                key = (track.track_id, zone.name)
                inside = zone.polygon.contains(point)

                if zone.zone_type == "intrusion":
                    events.extend(self._check_intrusion(track, zone, key, inside, frame_number))
                elif zone.zone_type == "loitering":
                    events.extend(self._check_loitering(track, zone, key, inside, frame_number))

                if not inside and key in self._zone_entry_frame:
                    del self._zone_entry_frame[key]  # left zone — reset timer for re-entry

        return events

    def _check_intrusion(self, track, zone, key, inside, frame_number):
        alert_key = (*key, "intrusion")
        timestamp = frame_number / self.fps

        if inside and alert_key not in self._alerted:
            self._alerted.add(alert_key)
            self._alert_fired_frame[alert_key] = frame_number
            return [Event(frame_number, timestamp, track.track_id, track.bbox,
                           "intrusion", zone.name, track.confidence)]
        if not inside and alert_key in self._alerted:
            self._alerted.discard(alert_key)  # allow re-alert on next entry
        return []

    def _check_loitering(self, track, zone, key, inside, frame_number):
        alert_key = (*key, "loitering")
        timestamp = frame_number / self.fps

        if not inside:
            return []

        if key not in self._zone_entry_frame:
            self._zone_entry_frame[key] = frame_number
            return []

        elapsed = (frame_number - self._zone_entry_frame[key]) / self.fps
        if elapsed >= zone.loiter_threshold_seconds and alert_key not in self._alerted:
            self._alerted.add(alert_key)
            self._alert_fired_frame[alert_key] = frame_number
            return [Event(frame_number, timestamp, track.track_id, track.bbox,
                           "loitering", zone.name, track.confidence)]
        return []

    def get_status(self, frame_number: int, tracks: list) -> Dict[int, List[ZoneStatus]]:
        """Live per-track zone occupancy for drawing on the annotated video.
        Reports CURRENT state every frame, unlike update() which only
        returns events on the frame they first trigger."""
        status: Dict[int, List[ZoneStatus]] = {}

        for track in tracks:
            point = self._bbox_center(track.bbox)
            for zone in self.zones:
                if not zone.polygon.contains(point):
                    continue

                key = (track.track_id, zone.name)
                elapsed = 0.0
                if zone.zone_type == "loitering" and key in self._zone_entry_frame:
                    elapsed = (frame_number - self._zone_entry_frame[key]) / self.fps

                just_triggered = False
                for event_type in ("intrusion", "loitering"):
                    alert_key = (*key, event_type)
                    fired_at = self._alert_fired_frame.get(alert_key)
                    if fired_at is not None and (frame_number - fired_at) <= ALERT_FLASH_FRAMES:
                        just_triggered = True

                status.setdefault(track.track_id, []).append(ZoneStatus(
                    zone_name=zone.name,
                    zone_type=zone.zone_type,
                    elapsed_seconds=elapsed,
                    threshold_seconds=zone.loiter_threshold_seconds,
                    just_triggered=just_triggered,
                ))

        return status