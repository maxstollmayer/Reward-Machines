from typing import Any, override

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.world_object import Door, Key
from minigrid.envs.doorkey import DoorKeyEnv

from utils import Action, Observation, Props, Reward


class DoorKey(DoorKeyEnv):
    def __init__(
        self, size: int = 5, max_steps: int | None = 250, **kwargs: Any
    ) -> None:
        super().__init__(size, max_steps, **kwargs)
        self.n_actions: int = self.action_space.n

    def door_is_open(self) -> bool:
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Door):
                    return cell.is_open
        raise ValueError(
            "ERROR: Unreachable! Door should always be generated by DoorKeyEnv._gen_grid."
        )

    def get_obs(self) -> Observation:
        obs = np.delete(self.grid.encode(), 1, axis=2)
        obs[self.agent_pos][0] = OBJECT_TO_IDX["agent"]
        obs[self.agent_pos][1] = self.agent_dir
        obs = obs[1:-1, 1:-1, :]
        return obs.reshape(obs.size)

    def get_props(self, terminated: bool, truncated: bool) -> Props:
        return {
            "door": self.door_is_open(),
            "key": isinstance(self.carrying, Key),
            "terminated": terminated,
            "truncated": truncated,
        }

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, Props]:
        _ = super().reset(seed=seed, options=options)
        obs = self.get_obs()
        props = {
            "door": False,
            "key": False,
            "terminated": False,
            "truncated": False,
        }

        return obs, props

    @override
    def step(self, action: Action) -> tuple[Observation, Reward, bool, bool, Props]:
        _, _, terminated, truncated, _ = super().step(action)
        obs = self.get_obs()
        reward = 1.0 if terminated else 0.0
        return (
            obs,
            reward,
            terminated,
            truncated,
            self.get_props(terminated, truncated),
        )

    @override
    def render(self) -> None:
        import pygame

        rgb_array = self.get_frame(highlight=False, agent_pov=False)
        rgb_array = np.transpose(rgb_array, axes=(1, 0, 2))

        if self.window is None:
            _ = pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(rgb_array.shape[:2])

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(rgb_array)
        _ = self.window.blit(surf, (0, 0))
        pygame.event.pump()
        _ = self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
