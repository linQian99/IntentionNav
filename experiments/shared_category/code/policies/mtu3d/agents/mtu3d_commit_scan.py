"""Three charged turns before revisiting a proposed object commitment."""
from __future__ import annotations


class CommitScan:
    """Retain four acquired views; never render or consume an action here."""

    def __init__(self, mode: str):
        if mode not in {'off', 'current', 'multiview'}:
            raise ValueError(f'Unknown commitment scan mode: {mode}')
        self.mode = mode
        self.views: list[dict] = []

    @property
    def active(self) -> bool:
        return bool(self.views)

    @property
    def ready(self) -> bool:
        return len(self.views) == 4

    def begin(self, observation: dict) -> None:
        if self.mode == 'off' or self.active:
            raise RuntimeError('Cannot start this commitment scan')
        self.views = [dict(observation)]

    def observe(self, observation: dict) -> None:
        if not self.active or self.ready:
            raise RuntimeError('Scan is not awaiting an observation')
        if any(v['observation'] == observation['observation'] for v in self.views):
            raise ValueError('A scan requires distinct acquired observations')
        self.views.append(dict(observation))

    def context(self) -> list[dict]:
        if not self.ready:
            raise RuntimeError('Three charged turns have not completed')
        return list(self.views[:-1]) if self.mode == 'multiview' else []

    def clear(self) -> None:
        self.views = []
