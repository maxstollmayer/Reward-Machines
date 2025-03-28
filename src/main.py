import minigrid

from agent import BLLearner, RMLearner
from envs.doorkey import RMDoorKey
from rm import RM
from train import test_BLLearner, test_RMLearner, train_BLLearner, train_RMLearner

# just so import does not get removed from formatter
minigrid.__version__

ENV_SIZE = 5
MAX_STEPS = 300
EPISODES = 1000
VERBOSITY = 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01


def main() -> None:
    env = RMDoorKey(size=ENV_SIZE, max_steps=MAX_STEPS, render_mode="rgb_array")
    rm = RM.from_file("src/envs/doorkey2.txt")

    print("Training Baseline:")
    bllearner = BLLearner(
        n_actions=env.action_space.n,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
    )
    _ = train_BLLearner(
        bllearner, env, rm, verbose=bool(VERBOSITY), report_each=VERBOSITY
    )
    _ = test_BLLearner(
        bllearner, env, rm, verbose=bool(VERBOSITY), render=bool(VERBOSITY)
    )

    print("Training using CRM:")
    rmlearner = RMLearner(
        n_actions=env.action_space.n,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
    )
    _ = train_RMLearner(
        rmlearner, env, rm, verbose=bool(VERBOSITY), report_each=VERBOSITY
    )
    _ = test_RMLearner(
        rmlearner, env, rm, verbose=bool(VERBOSITY), render=bool(VERBOSITY)
    )


if __name__ == "__main__":
    main()
