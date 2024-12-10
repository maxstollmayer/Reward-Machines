from gymnasium import Env
from gymnasium.core import ActType, ObsType


class RMMixin:
    def get_events(self) -> dict[str, bool]: ...


class RMEnv(Env[ObsType, ActType], RMMixin):
    pass
