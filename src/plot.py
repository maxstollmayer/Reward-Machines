import argparse
import pickle

import numpy as np
import scienceplots
from matplotlib import pyplot as plt

scienceplots.__path__
plt.style.use(["science", "bright"])

ALGS = ["q", "crm"]

datarun = tuple[list[float], list[float], list[int]]
dataset = dict[str, list[datarun]]


def get_args() -> tuple[int, int, int, str]:
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
    return (args.runs, args.size, args.end, args.folder)


def get_data(folder: str, runs: int, size: int) -> dataset:
    data: dataset = {alg: list() for alg in ALGS}
    for alg in ALGS:
        for i in range(runs):
            with open(f"{folder}/{alg}_{size}x{size}_{i+1}.pkl", "rb") as file:
                loaded = pickle.load(file)
            data[alg].append(loaded)
    return data


def get_avgs(data: dataset, runs: int) -> dict[str, datarun]:
    return {alg: np.sum(dataruns, axis=0) / runs for alg, dataruns in data.items()}


def plot(data: dict[str, datarun], runs: int, size: int, end: int) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Training error per episode")
    axs[1].set_title("Reward per episode")
    axs[2].set_title("Episode length")
    for i, (alg, arr) in enumerate(data.items()):
        axs[0].plot(arr[0][:end], color=f"C{i}")
        axs[1].plot(arr[1][:end], color=f"C{i}")
        axs[2].plot(arr[2][:end], color=f"C{i}", label=alg)
    axs[2].legend()
    fig.suptitle(f"Averages over {runs} runs")
    fig.savefig(f"figures/fig_{runs}runs_{size}x{size}.png")
    fig.savefig(f"figures/fig_{runs}runs_{size}x{size}.pgf")
    fig.savefig(f"figures/fig_{runs}runs_{size}x{size}.pdf")


def main() -> None:
    runs, size, end, folder = get_args()
    data = get_data(folder, runs, size)
    avgs = get_avgs(data, runs)
    plot(avgs, runs, size, end)


if __name__ == "__main__":
    main()
