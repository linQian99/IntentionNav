"""Keep only already acquired, sequential RGB-D views for the MTU policy.

This helper never renders, chooses a target, changes a score or acts. The
controller pays three initial turns; later calls reuse recent acquired views.
"""
from __future__ import annotations
import copy


class AcquiredHistory:
    """Bounded four-view history in acquisition order, including current view."""
    def __init__(self):
        self.views: list[tuple[int, dict]] = []
        self.last_step = 0
        self._seen: set[str] = set()

    def observe(self, view: dict, step: int) -> None:
        if step != self.last_step + 1:
            raise ValueError('History requires one acquisition per consecutive paid-action step')
        if set(view) != {'observation', 'position', 'look_dir', 'intrinsics'}:
            raise ValueError('History accepts only acquired view metadata')
        path = view['observation']
        if not isinstance(path, str) or not path or path in self._seen:
            raise ValueError('Each history observation must have a distinct saved identity')
        self._seen.add(path)
        self.views.append((step, copy.deepcopy(view)))
        self.views = self.views[-4:]
        self.last_step = step

    def context(self) -> list[dict]:
        if len(self.views) != 4:
            raise RuntimeError('Initial paid coverage is incomplete')
        return [copy.deepcopy(view) for _, view in self.views[:-1]]

    @property
    def steps(self) -> list[int]:
        return [step for step, _ in self.views]
