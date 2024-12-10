import hashlib
import pickle

import gymnasium as gym
import minigrid
from matplotlib import pyplot as plt

from agent import QLearner

minigrid.__version__  # just so import does not get removed

# make rm and rmwrapper class, see rm.py
# add "get_events" method to relevant environments from minigrid

# train agent using q learning for reward machines
# question: if i give the propositions to the qlearner does that speed up the training?
# or is there something more to the rm structure than the extra bit of state?

IDX_TO_ACTION = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


def observation_to_state(observation) -> int:
    dir = observation["direction"]
    img = observation["image"].tolist()
    string = str(dir) + " " + str(img)
    hashed = hashlib.md5(string.encode("utf-8"))
    return int(hashed.hexdigest(), 16)


def train_agent_on_minigrid(
    agent: QLearner, env_name: str, episodes: int = 1000
) -> tuple[list[float], list[int]]:
    env = gym.make(env_name)
    episode_rewards = []
    episode_lengths = []
    for episode in range(episodes):
        observation, _ = env.reset()
        state = observation_to_state(observation)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.get_action(state, explore=True)

            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state(next_observation)
            agent.update(state, action, float(reward), next_state)

            state = next_state
            total_reward += float(reward)
            steps += 1

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if (episode + 1) % 100 == 0:
            print(
                f"episode {episode+1}: total reward = {total_reward}, steps = {steps}, epsilon = {agent.epsilon}"
            )

    env.close()
    return episode_rewards, episode_lengths


def test_agent(agent: QLearner, env_name: str) -> None:
    env = gym.make(env_name, render_mode="human")
    observation, _ = env.reset(seed=0)
    state = observation_to_state(observation)
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.get_action(state, explore=False)
        print(IDX_TO_ACTION[action])
        observation, reward, terminated, truncated, _ = env.step(action)
        state = observation_to_state(observation)
        env.render()

    env.close()


if __name__ == "__main__":
    # with open("weights.pkl", "rb") as file:
    #     weights: dict[int, list[float]] = pickle.load(file)

    agent = QLearner(n_actions=7)
    rewards, steps = train_agent_on_minigrid(
        agent, "MiniGrid-DoorKey-5x5-v0", episodes=2000
    )
    with open("weights.pkl", "wb") as file:
        pickle.dump(agent.weights, file)

    plt.plot(rewards, label="total reward per episode")
    plt.xlabel("episode")
    plt.show()
    plt.plot(steps, label="length of episode")
    plt.xlabel("episode")
    plt.show()

    test_agent(agent, "MiniGrid-DoorKey-5x5-v0")
