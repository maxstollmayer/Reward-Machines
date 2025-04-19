from agent import DQNAgent
from envs.doorkey import DoorKey
from rm import RM
from train import test, train

ENV_SIZE = 5
STATE_DIM = 1 + 2 * (ENV_SIZE - 2) ** 2
HIDDEN_DIM = 256
MAX_STEPS = 250
SEED = 0
EPISODES = 1000
VERBOSITY = 100
ALPHA = 1e-4
GAMMA = 0.9
EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY: float = (MIN_EPSILON / EPSILON) ** (1 / EPISODES)


def main() -> None:
    env = DoorKey(size=ENV_SIZE, max_steps=MAX_STEPS)
    rm = None
    rm = RM.from_file("src/envs/doorkey.txt")
    agent = DQNAgent(
        n_actions=env.n_actions,
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
    )

    print("Started training:")
    _ = train(
        agent,
        env,
        rm,
        episodes=EPISODES,
        verbose=bool(VERBOSITY),
        report_each=VERBOSITY,
    )

    print("Finished training. Starting test:")
    steps = test(
        agent, env, rm, verbose=bool(VERBOSITY), render=bool(VERBOSITY), seed=SEED
    )
    print(f"Finished test: took {steps} steps.")


if __name__ == "__main__":
    main()
