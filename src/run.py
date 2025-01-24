import argparse
import pickle

from agent import QLearner, RMLearner
from envs.doorkey import RMDoorKey
from rm import RM
from train import train_QLearner, train_RMLearner

VERBOSE = False
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01


def get_args() -> tuple[str, int, int, int, str, str]:
    parser = argparse.ArgumentParser(
        description="Run the experiment once using different parameters."
    )
    _ = parser.add_argument(
        "algorithm", choices=["q", "crm"], type=str, help="algorithm to use"
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
    return (
        args.algorithm,
        args.size,
        args.max_steps,
        args.episodes,
        args.suffix,
        args.folder,
    )


def run_q(
    size: int, max_steps: int, episodes: int
) -> tuple[list[float], list[float], list[int]]:
    env = RMDoorKey(size=size, max_steps=max_steps)
    learner = QLearner(
        n_actions=env.action_space.n,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
    )
    return train_QLearner(learner, env, episodes=episodes, verbose=VERBOSE)


def run_rm(
    size: int, max_steps: int, episodes: int
) -> tuple[list[float], list[float], list[int]]:
    env = RMDoorKey(size=size, max_steps=max_steps)
    rm = RM.from_file("doorkey.txt")
    learner = RMLearner(
        n_actions=env.action_space.n,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
    )
    return train_RMLearner(learner, env, rm, episodes=episodes, verbose=VERBOSE)


def save_data(data: tuple[list[float], list[float], list[int]], filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def main() -> None:
    alg, size, max_steps, episodes, suffix, folder = get_args()

    if alg == "q":
        data = run_q(size, max_steps, episodes)
    else:
        data = run_rm(size, max_steps, episodes)

    save_data(data, f"{folder}/{alg}_{size}x{size}_{suffix}.pkl")


if __name__ == "__main__":
    main()
