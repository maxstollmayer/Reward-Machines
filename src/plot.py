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


ALGS = {
    "q": ("C0", "Q-Learning"),
    "bl": ("C1", "Baseline"),
    "crm": ("C2", "CRM"),
    "bl2": ("C3", "Baseline tuned"),
    "crm2": ("C4", "CRM tuned"),
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
    test_steps: list[int]


Data = dict[str, list[Run]]
Avgs = dict[str, Avg]


class Args(NamedTuple):
    runs: int
    size: int
    folder: str


def get_args() -> Args:
    parser = argparse.ArgumentParser(description="Plot data for given experiments")
    _ = parser.add_argument(
        "-s", "--size", type=int, default=5, help="size of the environment (default 5x5"
    )
    _ = parser.add_argument(
        "-n", "--runs", type=int, default=100, help="number of runs"
    )
    _ = parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="data",
        help="folder that the data is stored in",
    )
    args = parser.parse_args()
    return Args(args.runs, args.size, args.folder)


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
    k = list(ALGS.keys())[0]
    n_runs = len(data[k])
    dir: Avgs = dict()
    for alg, runs in data.items():
        errors: list[float] = np.sum([run.errors for run in runs], axis=0) / n_runs
        rewards: list[float] = np.sum([run.rewards for run in runs], axis=0) / n_runs
        steps: list[float] = np.sum([run.steps for run in runs], axis=0) / n_runs
        dir[alg] = Avg(errors, rewards, steps, [run.test_steps for run in runs])
    return dir


def print_result(avgs: Avgs, runs: int) -> None:
    k = list(ALGS.keys())[0]
    episodes = len(avgs[k][0])
    n = max(len(alg) for alg in ALGS)
    print(
        f"Number of steps in {runs} runs of exploiting after training for {episodes} episodes:"
    )
    print(" | ".join([" " * n, "best", "count", "median", " mean", "  std"]))
    for alg, avg in avgs.items():
        best_steps = np.min(avg.test_steps)
        best_count = len([x for x in avg.test_steps if x == best_steps])
        median_steps = int(np.median(avg.test_steps))
        mean_steps = np.mean(avg.test_steps)
        std_steps = np.std(avg.test_steps)
        print(
            f"{alg.rjust(n)} | {best_steps:4} | {best_count:5} | {median_steps:6} | {mean_steps:5} | {std_steps:5.2f}"
        )


def plot_comparison1(avgs: Avgs, runs: int, size: int) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(6.27, 1.425))
    for alg, avg in ((k, avgs[k]) for k in ["q", "bl", "crm"]):
        color, label = ALGS[alg]
        axs[0].plot(avg.errors, color=color, alpha=0.5, label=label)
        axs[1].plot(avg.rewards, color=color, alpha=0.5)
        axs[2].plot(avg.steps, color=color, alpha=0.5)
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[2].set_xlabel("Episode")
    axs[0].set_title("Training (TD) error")
    axs[1].set_title("Total reward")
    axs[2].set_title("Steps")
    axs[0].legend()
    fig.savefig(f"paper/figures/cmp1_{runs}runs_{size}x{size}.pdf", format="pdf")


def plot_comparison2(avgs: Avgs, runs: int, size: int) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(6.27, 1.425))
    for alg, avg in ((k, avgs[k]) for k in ["bl2", "crm2"]):
        color, label = ALGS[alg]
        axs[0].plot(avg.errors, color=color, alpha=0.5, label=label)
        axs[1].plot(avg.rewards, color=color, alpha=0.5)
        axs[2].plot(avg.steps, color=color, alpha=0.5)
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[2].set_xlabel("Episode")
    axs[0].set_title("Training (TD) error")
    axs[1].set_title("Total reward")
    axs[2].set_title("Steps")
    axs[0].legend()
    fig.savefig(f"paper/figures/cmp2_{runs}runs_{size}x{size}.pdf", format="pdf")


def plot_comparison3(avgs: Avgs, runs: int, size: int) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(4.2, 1.425))
    for alg, avg in avgs.items():
        color, label = ALGS[alg]
        axs[0].plot(avg.errors, color=color, alpha=0.5, label=label)
        axs[1].plot(avg.steps, color=color, alpha=0.5, label=label)
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[0].set_title("Training (TD) error")
    axs[1].set_title("Steps")
    axs[1].legend(loc="upper right", bbox_to_anchor=(2.1, 1), ncols=1)
    fig.savefig(f"paper/figures/cmp3_{runs}runs_{size}x{size}.pdf", format="pdf")


def main() -> None:
    runs, size, folder = get_args()
    data = get_data(folder, runs, size)
    avgs = get_avgs(data)
    print_result(avgs, runs)
    plot_comparison1(avgs, runs, size)
    plot_comparison2(avgs, runs, size)
    plot_comparison3(avgs, runs, size)


if __name__ == "__main__":
    main()
