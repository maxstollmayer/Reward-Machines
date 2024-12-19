import minigrid
import numpy as np
from matplotlib import pyplot as plt

from agent import QLearner, RMLearner
from envs.doorkey import RMDoorKey
from rm import RM
from train import train_QLearner, train_RMLearner

minigrid.__version__  # just so import does not get removed from formatter

EPISODES = 1000
ROLLING_LENGTH = 100


def plot_compact(
    data: dict[str, tuple[list[float], list[float], list[int]]],
    rolling_length: int = 100,
) -> None:
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    for name, (errors, rewards, steps) in data.items():
        errors = (
            np.convolve(errors, np.ones(rolling_length), mode="same") / rolling_length
        )
        rewards = (
            np.convolve(rewards, np.ones(rolling_length), mode="valid") / rolling_length
        )
        steps = (
            np.convolve(steps, np.ones(rolling_length), mode="same") / rolling_length
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


if __name__ == "__main__":
    env = RMDoorKey(render_mode="rgb_array")
    rm = RM.from_file("doorkey.txt")

    qlearner = QLearner(n_actions=env.action_space.n, epsilon=1, epsilon_decay=0.995)
    rmlearner = RMLearner(n_actions=env.action_space.n)

    print("Training Q Learner:")
    qerrors, qrewards, qsteps = train_QLearner(
        qlearner, env.unwrapped, episodes=EPISODES, report_each=ROLLING_LENGTH
    )
    print("\nTraining RM Learner:")
    rmerrors, rmrewards, rmsteps = train_RMLearner(
        rmlearner, env.unwrapped, rm, episodes=EPISODES, report_each=ROLLING_LENGTH
    )

    plot_compact(
        {
            "Q-learning": (qerrors, qrewards, qsteps),
            "CRM": (rmerrors, rmrewards, rmsteps),
        },
        rolling_length=ROLLING_LENGTH,
    )

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
