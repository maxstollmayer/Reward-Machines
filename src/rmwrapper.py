from typing import Any, override

import numpy as np
from gymnasium import Wrapper, spaces
from numpy.typing import NDArray

from env import RMEnv
from rm import RM


class RMWrapper(Wrapper[Any, Any, Any, Any]):
    # https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/reward_machines/rm_environment.py

    def __init__(self, env: RMEnv[Any, Any], reward_machines: list[RM]) -> None:
        super().__init__(env)
        self.rms: list[RM] = reward_machines
        self.num_rm_states: int = sum([len(rm.get_states()) for rm in reward_machines])

        self.observation_dict: spaces.Dict = spaces.Dict(
            {
                "features": env.observation_space,
                "rm-state": spaces.Box(
                    low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8
                ),
            }
        )
        flatdim = spaces.flatdim(self.observation_dict)
        low: float = float(env.observation_space.low[0])
        high: float = float(env.observation_space.high[0])
        self.observation_space: spaces.Box = spaces.Box(
            low=low, high=high, shape=(flatdim,), dtype=np.float32
        )

        self.rm_state_features: dict[tuple[int, int], NDArray[np.float32]] = {}
        for rm_id, rm in enumerate(self.rms):
            for u_id in rm.get_states():
                u_features = np.zeros(self.num_rm_states, dtype=np.float32)
                u_features[len(self.rm_state_features)] = 1
                self.rm_state_features[(rm_id, u_id)] = u_features
        self.rm_done_feat: NDArray[np.float32] = np.zeros(
            self.num_rm_states, np.float32
        )

        self.current_rm_id: int = 1
        self.current_rm: RM | None = None

    @override
    def reset(self, seed: int, options: int) -> tuple[int]:
        self.obs = self.env.reset()
        self.current_rm_id = (self.current_rm_id + 1) % len(self.rms)
        self.current_rm = self.rms[self.current_rm_id]
        self.current_u_id = self.current_rm.reset()

        return self.get_observation(
            self.obs, self.current_rm_id, self.current_u_id, False
        )

    def get_observation(self, next_obs, rm_id: int, u_id: int, done: bool) -> None:
        rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id, u_id)]
        rm_obs = {"features": next_obs, "rm-state": rm_feat}
        return spaces.flatten(self.observation_dict, rm_obs)

    def step(self, action) -> None:
        next_obs, original_reward, terminated, truncated, info = self.env.step(action)

        # TODO: implement own Env type that has the method get_events and more restricted spaces
        props = self.env.get_events()
        # TODO: expect "terminated" and "truncated" to always be propositions? or just take them as parameters? probably the latter!
        self.crm_params = (
            self.obs,
            action,
            next_obs,
            terminated,
            truncated,
            props,
            info,
        )
        self.obs = next_obs

        self.current_u_id, rm_reward, rm_done = self.current_rm.step(
            self.current_u_id, props
        )

        done = rm_done or truncated or terminated
        rm_obs = self.get_observation(
            next_obs, self.current_rm_id, self.current_u_id, done
        )
        return rm_obs, rm_reward, done, info
