from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from objtracker.tracking.types import Detections, Track


class MultiObjectTracker(Protocol):
    """Common interface for online multi-object trackers."""

    def reset(self) -> None:
        """Clear all per-sequence state."""
        ...

    def update(
        self, detections: Detections | Mapping[str, Tensor], frame: Any | None = None
    ) -> list[Track]:
        """Advance the tracker by one frame and return current tracks."""
        ...
