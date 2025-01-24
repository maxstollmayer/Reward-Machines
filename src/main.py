import minigrid
import numpy as np
import scienceplots
from matplotlib import pyplot as plt
from tqdm import trange

from agent import QLearner, RMLearner
from envs.doorkey import RMDoorKey
from rm import RM
from train import train_QLearner, train_RMLearner

# just so import does not get removed from formatter
minigrid.__version__
scienceplots.__path__

plt.style.use(["science", "bright"])  # for scientific paper quality plots

NUM_RUNS = 10
ENV_SIZE = 6
MAX_STEPS = 300
EPISODES = 2021
ROLLING_LENGTH = EPISODES // 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01


def plot_compact(
    data: dict[str, tuple[list[float], list[float], list[int]]],
    rolling_length: int = 100,
) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    for name, (errors, rewards, steps) in data.items():
        errors = (
            np.convolve(errors, np.ones(rolling_length), mode="valid") / rolling_length
        )
        rewards = (
            np.convolve(rewards, np.ones(rolling_length), mode="valid") / rolling_length
        )
        steps = (
            np.convolve(steps, np.ones(rolling_length), mode="valid") / rolling_length
        )
        axs[0].set_title("Training error per episode")
        axs[0].plot(errors)
        axs[1].set_title("Total rewards per episode")
        axs[1].plot(rewards)
        axs[2].set_title("Episode length")
        axs[2].plot(steps, label=name)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_runs(
    data: dict[str, list[tuple[list[float], list[float], list[int]]]],
    rolling_length: int = 100,
) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Training error per episode")
    axs[1].set_title("Total rewards per episode")
    axs[2].set_title("Episode length")
    for i, (name, runs) in enumerate(data.items()):
        n = len(runs)
        m = len(runs[0][0]) - rolling_length + 1
        avg_errors = np.zeros(m)
        avg_rewards = np.zeros(m)
        avg_steps = np.zeros(m)
        for errors, rewards, steps in runs:
            rolled_errors = (
                np.convolve(errors, np.ones(rolling_length), mode="valid")
                / rolling_length
            )
            rolled_rewards = (
                np.convolve(rewards, np.ones(rolling_length), mode="valid")
                / rolling_length
            )
            rolled_steps = (
                np.convolve(steps, np.ones(rolling_length), mode="valid")
                / rolling_length
            )
            avg_errors += rolled_errors
            avg_rewards += rolled_rewards
            avg_steps += rolled_steps
            axs[0].plot(rolled_errors, color=f"C{i}", alpha=1 / NUM_RUNS)
            axs[1].plot(rolled_rewards, color=f"C{i}", alpha=1 / NUM_RUNS)
            axs[2].plot(rolled_steps, color=f"C{i}", alpha=1 / NUM_RUNS)

        avg_errors /= n
        avg_rewards /= n
        avg_steps /= n
        axs[0].plot(avg_errors, color=f"C{i}")
        axs[1].plot(avg_rewards, color=f"C{i}")
        axs[2].plot(avg_steps, color=f"C{i}", label=f"{name} (avg)")

    axs[2].legend()
    plt.tight_layout()
    fig.savefig(f"fig_{NUM_RUNS}runs_{ENV_SIZE}x{ENV_SIZE}.png")
    fig.savefig(f"fig_{NUM_RUNS}runs_{ENV_SIZE}x{ENV_SIZE}.pgf")


if __name__ == "__main__":
    data = {
        "Q-learning": [],
        "CRM": [],
    }

    env = RMDoorKey(size=ENV_SIZE, max_steps=MAX_STEPS, render_mode="rgb_array")
    rm = RM.from_file("doorkey.txt")

    for _ in trange(NUM_RUNS, desc="Q-learning"):
        qlearner = QLearner(
            n_actions=env.action_space.n,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            min_epsilon=MIN_EPSILON,
        )
        qdata = train_QLearner(qlearner, env, episodes=EPISODES, verbose=False)
        data["Q-learning"].append(qdata)

    for _ in trange(NUM_RUNS, desc="CRM"):
        rmlearner = RMLearner(
            n_actions=env.action_space.n,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            min_epsilon=MIN_EPSILON,
        )
        rmdata = train_RMLearner(rmlearner, env, rm, episodes=EPISODES, verbose=False)
        data["CRM"].append(rmdata)

    plot_runs(data, rolling_length=ROLLING_LENGTH)

    # print("Training Q Learner:")
    # qerrors, qrewards, qsteps = train_QLearner(
    #     qlearner, env, episodes=EPISODES, report_each=ROLLING_LENGTH
    # )
    # print("\nTraining RM Learner:")
    # rmerrors, rmrewards, rmsteps = train_RMLearner(
    #     rmlearner, env, rm, episodes=EPISODES, report_each=ROLLING_LENGTH
    # )
    #
    # plot_compact(
    #     {
    #         "Q-learning": (qerrors, qrewards, qsteps),
    #         "CRM": (rmerrors, rmrewards, rmsteps),
    #     },
    #     rolling_length=ROLLING_LENGTH,
    # )

    # plt.plot(index, qerrors, label="Q-learning")
    # plt.plot(index, rmerrors, label="CRM")
    # plt.xlabel("episode")
    # plt.ylabel("temporal difference errors")
    # plt.show()
    #
    # plt.plot(index, qrewards, label="Q-learning")
    # plt.plot(index, rmrewards, label="CRM")
    # plt.xlabel("episode")
    # plt.ylabel("total reward")
    # plt.show()
    #
    # plt.plot(index, qsteps, label="Q-learning")
    # plt.plot(index, rmsteps, label="CRM")
    # plt.xlabel("episode")
    # plt.ylabel("steps")
    # plt.show()

    # TODO: save figures using https://github.com/nschloe/tikzplotlib

    # test_Qlearner(qlearner, env)
