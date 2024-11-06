import gymnasium as gym
import minigrid

from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv

IDX_TO_ACTION = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIR_TO_STR = ["right", "down", "left", "up"]


def get_door(env: MiniGridEnv):
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            cell = env.grid.get(x, y)
            if isinstance(cell, Door):
                return cell
    raise ValueError("Unreachable: Door is always generated.")


env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
observation, info = env.reset(seed=0)
door = get_door(env.unwrapped)


episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("action:", IDX_TO_ACTION[action])
    print("direction:", DIR_TO_STR[observation["direction"]])
    print("carrying key:", env.unwrapped.carrying is not None)
    print("opened door", door.is_open)
    print(flush=True)
    episode_over = terminated or truncated

env.close()
