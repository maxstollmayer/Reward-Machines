from typing import Any, Protocol

from utils import Action, Observation, Props, Reward


class Env(Protocol):
    n_actions: int

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, Props]: ...
    def step(self, action: Action) -> tuple[Observation, Reward, bool, bool, Props]: ...
    def close(self) -> None: ...
    def render(self) -> None: ...
