import argparse
import json
from typing import NamedTuple

from agent import Agent, BLLearner, CRMLearner, QLearner
from env import RMMiniGridEnv
from envs.doorkey import RMDoorKey
from rm import RM
from train import test, train

VERBOSE = False
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01


class Run(NamedTuple):
    errors: list[float]
    rewards: list[float]
    steps: list[int]
    test_steps: int


class Args(NamedTuple):
    alg: str
    size: int
    max_steps: int
    episodes: int
    suffix: str
    folder: str


def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Run the experiment once using different parameters."
    )
    _ = parser.add_argument(
        "algorithm",
        choices=["q", "bl", "crm", "bl2", "crm2"],
        type=str,
        help="algorithm to use",
    )
    _ = parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=5,
        help="size of the doorkey environment (default: 5x5)",
    )
    _ = parser.add_argument(
        "-m",
        "--max-steps",
        type=int,
        default=300,
        help="maximum steps in each episode before terminating",
    )
    _ = parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=1000,
        help="number of episodes to run the algorithm",
    )
    _ = parser.add_argument(
        "-x", "--suffix", type=str, default="", help="suffix for data filename"
    )
    _ = parser.add_argument(
        "-f", "--folder", type=str, default="data", help="folder to store the data in"
    )
    args = parser.parse_args()
    return Args(
        args.algorithm,
        args.size,
        args.max_steps,
        args.episodes,
        args.suffix,
        args.folder,
    )


def run(
    agent: Agent[tuple[int, int], int] | Agent[int, int],
    env: RMMiniGridEnv,
    rm: RM | None = None,
    episodes: int = 1,
) -> Run:
    train_data = train(agent, env, rm, episodes, verbose=VERBOSE)
    test_data = test(agent, env, rm, verbose=VERBOSE)
    return Run(*train_data, test_data)


def save_data(data: Run, filename: str) -> None:
    with open(filename, "w") as file:
        json.dump(data, file)


def main() -> None:
    alg, size, max_steps, episodes, suffix, folder = get_args()

    env = RMDoorKey(size=size, max_steps=max_steps)

    match alg:
        case "q":
            agent = QLearner(
                n_actions=env.action_space.n,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY,
                min_epsilon=MIN_EPSILON,
            )
            data = run(agent, env, episodes=episodes)
        case "bl":
            rm = RM.from_file("src/envs/doorkey.txt")
            agent = BLLearner(
                n_actions=env.action_space.n,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY,
                min_epsilon=MIN_EPSILON,
            )
            data = run(agent, env, rm, episodes)
        case "crm":
            rm = RM.from_file("src/envs/doorkey.txt")
            agent = CRMLearner(
                n_actions=env.action_space.n,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY,
                min_epsilon=MIN_EPSILON,
            )
            data = run(agent, env, rm, episodes)
        case "bl2":
            rm = RM.from_file("src/envs/doorkey2.txt")
            agent = BLLearner(
                n_actions=env.action_space.n,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY,
                min_epsilon=MIN_EPSILON,
            )
            data = run(agent, env, rm, episodes)
        case "crm2":
            rm = RM.from_file("src/envs/doorkey2.txt")
            agent = CRMLearner(
                n_actions=env.action_space.n,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY,
                min_epsilon=MIN_EPSILON,
            )
            data = run(agent, env, rm, episodes)
        case _:
            raise ValueError(f"ERROR: unknown algorithm '{alg}'.")

    save_data(data, f"{folder}/{alg}_{size}x{size}_{suffix}.json")


if __name__ == "__main__":
    main()
