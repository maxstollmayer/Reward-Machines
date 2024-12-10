from typing import Any, override

from minigrid.core.world_object import Door, Key
from minigrid.envs.doorkey import DoorKeyEnv

from env import RMMixin


class RMDoorKey(DoorKeyEnv, RMMixin):
    def __init__(
        self, size: int = 5, max_steps: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(size, max_steps, **kwargs)
        self.door: Door = self.get_door()
        self.props: dict[str, bool] = {
            "door": False,
            "key": False,
            "truncated": False,
            "terminated": False,
        }

    def get_door(self) -> Door:
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Door):
                    return cell
        raise ValueError("Unreachable: Door and key are always generated.")

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        val = super().reset(seed=seed, options=options)
        self.door = self.get_door()
        self.props = {
            "door": False,
            "key": False,
            "truncated": False,
            "terminated": False,
        }
        return val

    def get_events(self) -> dict[str, bool]:
        self.props["door"] = self.door.is_open
        self.props["key"] = isinstance(self.carrying, Key)
        # TODO: let these be handled by the wrapper? truncated always bad and terminated always good?
        self.props["truncated"] = ...
        self.props["terminated"] = ...
        return self.props
