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
    tests: int


Data = dict[str, list[Run]]


class Avg(NamedTuple):
    errors: list[float]
    rewards: list[float]
    steps: list[int]
    tests: list[int]


Avgs = dict[str, Avg]


class Args(NamedTuple):
    runs: int
    size: int
    folder: str


def get_args() -> Args:
    parser = argparse.ArgumentParser(description="Plot data for given experiments")
    _ = parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=5,
        help="size of the environment (default 5x5)",
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
    dir: Avgs = dict()
    for alg, runs in data.items():
        errors: list[float] = np.mean([run.errors for run in runs], axis=0)
        rewards: list[float] = np.mean([run.rewards for run in runs], axis=0)
        steps: list[int] = np.mean([run.steps for run in runs], axis=0)
        dir[alg] = Avg(errors, rewards, steps, [run.tests for run in runs])
    return dir


def calc_convergence(avg: Avg) -> int:
    # TODO: better measure?
    for i, error in enumerate(avg.errors):
        if error <= 1:
            return i
    return -1


def print_result(avgs: Avgs, runs: int) -> None:
    episodes = len(avgs["q"].errors)
    n = max(len(alg) for alg in ALGS)
    print(
        f"Number of steps in {runs} runs of exploiting after training for {episodes} episodes:"
    )
    print(" | ".join([" " * n, "convergence", "best", "count", "q25", "q50", "q75"]))
    for alg, avg in avgs.items():
        converged_at = calc_convergence(avg)
        best_steps: int = np.min(avg.tests)
        best_count = len([x for x in avg.tests if x == best_steps]) / runs * 100
        q25 = np.quantile(avg.tests, 0.25)
        q50 = np.quantile(avg.tests, 0.5)
        q75 = np.quantile(avg.tests, 0.75)
        print(
            f"{alg.rjust(n)} | {converged_at:11} | {best_steps:4} | {best_count:4.0f}% | {q25:3.0f} | {q50:3.0f} | {q75:3.0f}"
        )


def plot_comparison1(avgs: Avgs, size: int) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(6.27, 1.425))
    for alg, avg in ((k, avgs[k]) for k in ["q", "bl", "crm"]):
        color, label = ALGS[alg]
        axs[0].plot(avg.errors, color=color, alpha=0.75, label=label)
        axs[1].plot(avg.rewards, color=color, alpha=0.75)
        axs[2].plot(avg.steps, color=color, alpha=0.75)
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[2].set_xlabel("Episode")
    axs[0].set_title("Total error")
    axs[1].set_title("Total reward")
    axs[2].set_title("Steps")
    axs[0].legend()
    fig.savefig(f"paper/figures/cmp1_{size}x{size}.pdf", format="pdf")


def plot_comparison2(avgs: Avgs, size: int) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(6.27, 1.425))
    for alg, avg in ((k, avgs[k]) for k in ["bl2", "crm2"]):
        color, label = ALGS[alg]
        axs[0].plot(avg.errors, color=color, alpha=0.75, label=label)
        axs[1].plot(avg.rewards, color=color, alpha=0.75)
        axs[2].plot(avg.steps, color=color, alpha=0.75)
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[2].set_xlabel("Episode")
    axs[0].set_title("Total error")
    axs[1].set_title("Total reward")
    axs[2].set_title("Steps")
    axs[0].legend()
    fig.savefig(f"paper/figures/cmp2_{size}x{size}.pdf", format="pdf")


def plot_comparison3(avgs: Avgs, size: int) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(4.2, 1.425))
    for alg, avg in avgs.items():
        color, label = ALGS[alg]
        axs[0].plot(avg.errors, color=color, alpha=0.75, label=label)
        axs[1].plot(avg.steps, color=color, alpha=0.75, label=label)
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[0].set_title("Total error")
    axs[1].set_title("Steps")
    axs[1].legend(loc="upper right", bbox_to_anchor=(2.1, 1), ncols=1)
    fig.savefig(f"paper/figures/cmp3_{size}x{size}.pdf", format="pdf")


def plot_comparison4(avgs: Avgs, size: int) -> None:
    fig, ax = plt.subplots(figsize=(2.1, 1.425))
    data = [avg.tests for avg in avgs.values()]
    parts = ax.violinplot(
        data,
        showextrema=False,
        showmeans=False,
        showmedians=False,
    )
    for pc, (color, _) in zip(parts["bodies"], ALGS.values()):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    ax.scatter(range(1, len(ALGS) + 1), np.mean(data, axis=1), color="k", marker="_")
    ax.set_xticks(
        range(1, len(ALGS) + 1),
        labels=[name for _, name in ALGS.values()],
        rotation="vertical",
    )
    ax.set_title("Testing steps distribution")
    fig.savefig(f"paper/figures/cmp4_{size}x{size}.pdf", format="pdf")


def main() -> None:
    runs, size, folder = get_args()
    data = get_data(folder, runs, size)
    avgs = get_avgs(data)
    plot_comparison1(avgs, size)
    plot_comparison2(avgs, size)
    plot_comparison3(avgs, size)
    plot_comparison4(avgs, size)
    print_result(avgs, runs)


if __name__ == "__main__":
    main()
