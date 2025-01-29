import argparse
import json
from typing import NamedTuple

import numpy as np
import scienceplots
from matplotlib import pyplot as plt

scienceplots.__path__
plt.style.use(["science", "bright"])
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 10,
        "pgf.rcfonts": False,
    }
)


ALGS = ["q", "crm"]
ALG_TO_LABEL = {
    "q": "Q-Learning",
    "crm": "CRM",
}


class Run(NamedTuple):
    errors: list[float]
    rewards: list[float]
    steps: list[int]
    test_steps: int


class Avg(NamedTuple):
    errors: list[float]
    rewards: list[float]
    steps: list[float]
    test_steps: float


Data = dict[str, list[Run]]
Avgs = dict[str, Avg]


class Args(NamedTuple):
    runs: int
    size: int
    end: int
    folder: str


def get_args() -> Args:
    parser = argparse.ArgumentParser(description="Plot data for given experiments")
    _ = parser.add_argument("-n", "--runs", type=int, default=10, help="number of runs")
    _ = parser.add_argument(
        "-s", "--size", type=int, default=5, help="size of the environment (default 5x5"
    )
    _ = parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=-1,
        help="plot data only until this value",
    )
    _ = parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="data",
        help="folder that the data is stored in",
    )
    args = parser.parse_args()
    return Args(args.runs, args.size, args.end, args.folder)


def get_data(folder: str, runs: int, size: int) -> Data:
    data: Data = {alg: list() for alg in ALGS}
    for alg in ALGS:
        for i in range(runs):
            with open(f"{folder}/{alg}_{size}x{size}_{i+1}.json", "r") as file:
                loaded: tuple[list[float], list[float], list[int], int] = json.load(
                    file
                )
            data[alg].append(Run(*loaded))
    return data


def get_avgs(
    data: Data,
) -> Avgs:
    n_runs = len(data[ALGS[0]])
    dir: Avgs = dict()
    for alg, runs in data.items():
        errors: list[float] = np.sum([run.errors for run in runs], axis=0) / n_runs
        rewards: list[float] = np.sum([run.rewards for run in runs], axis=0) / n_runs
        steps: list[float] = np.sum([run.steps for run in runs], axis=0) / n_runs
        test_steps = sum([run.test_steps for run in runs]) / n_runs
        dir[alg] = Avg(errors, rewards, steps, test_steps)
    return dir


def print_result(avgs: Avgs, runs: int) -> None:
    episodes = len(avgs[ALGS[0]][0])
    n = max(len(alg) for alg in ALGS)
    print(
        f"{runs} run average of steps of exploiting after training for {episodes} episodes:"
    )
    for alg, run in avgs.items():
        print(f"{alg.ljust(n)}: {run.test_steps}")


def plot(avgs: Avgs, runs: int, size: int, end: int) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(6.27, 1.425))
    for i, (alg, avg) in enumerate(avgs.items()):
        axs[0].plot(avg.errors[:end], color=f"C{i}", label=ALG_TO_LABEL[alg])
        axs[1].plot(avg.rewards[:end], color=f"C{i}")
        axs[2].plot(avg.steps[:end], color=f"C{i}")
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[2].set_xlabel("Episode")
    axs[0].set_title("Training (TD) error")
    axs[1].set_title("Total reward")
    axs[2].set_title("Steps")
    axs[0].legend()
    fig.savefig(f"paper/figures/fig_{runs}runs_{size}x{size}.pdf", format="pdf")


def main() -> None:
    runs, size, end, folder = get_args()
    data = get_data(folder, runs, size)
    avgs = get_avgs(data)
    print_result(avgs, runs)
    plot(avgs, runs, size, end)


if __name__ == "__main__":
    main()
