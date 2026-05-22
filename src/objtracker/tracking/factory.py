from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from objtracker.tracking.bot_sort_track import BoTSORTTrack, BoTSORTTrackConfig
from objtracker.tracking.byte_track import ByteTrack, ByteTrackConfig

if TYPE_CHECKING:
    from objtracker.tracking.base import MultiObjectTracker


@dataclass(frozen=True)
class TrackerRegistration:
    tracker_cls: type
    config_cls: type


TRACKERS = {
    "botsort": TrackerRegistration(BoTSORTTrack, BoTSORTTrackConfig),
    "bytetrack": TrackerRegistration(ByteTrack, ByteTrackConfig),
}


def build_tracker(name: str, **kwargs: Any) -> MultiObjectTracker:
    registration = TRACKERS.get(name)
    if registration is None:
        msg = f"Unsupported tracker: {name}"
        raise ValueError(msg)

    config = kwargs.pop("config", None)
    if config is not None and kwargs:
        msg = "Pass either config or tracker config keyword fields, not both"
        raise ValueError(msg)
    return registration.tracker_cls(config=config or registration.config_cls(**kwargs))
