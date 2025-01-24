import numpy as np
from gymnasium.wrappers import HumanRendering

from agent import QLearner, RMLearner
from env import RMMiniGridEnv
from rm import RM

IDX_TO_ACTION = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


def train_QLearner(
    agent: QLearner,
    env: RMMiniGridEnv,
    episodes: int = 1000,
    report_each: int = 100,
    verbose: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    td_errors = []
    total_rewards = []
    steps = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        length = 0

        while not done:
            action = agent.get_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, done, next_state)
            state = next_state
            total_reward += reward
            length += 1

        td_error = np.sum(agent.td_errors[-length:]) / length
        td_errors.append(td_error)
        total_rewards.append(total_reward)
        steps.append(length)

        if episode % report_each == 0:
            if verbose:
                print(
                    f"episode {episode:4}: error={td_error: .3f}, reward={total_reward:.3f}, steps={length:3.0f}, epsilon={agent.epsilon:.3f}"
                )

        agent.decay_epsilon()

    env.close()
    return td_errors, total_rewards, steps


def test_Qlearner(agent: QLearner, env: RMMiniGridEnv) -> None:
    env = HumanRendering(env)
    state, _ = env.reset(seed=0)
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.get_action(state, explore=False)
        print(IDX_TO_ACTION[action])
        state, _, terminated, truncated, _ = env.step(action)
        env.render()

    env.close()


def train_RMLearner(
    agent: RMLearner,
    env: RMMiniGridEnv,
    rm: RM,
    episodes: int = 1000,
    report_each: int = 100,
    verbose: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    td_errors = []
    total_rewards = []
    steps = []
    for episode in range(episodes):
        env_state, _ = env.reset()
        u = rm.reset()
        state = (env_state, u)
        done = False
        total_reward = 0
        length = 0

        while not done:
            action = agent.get_action(state, explore=True)

            next_env_state, _reward, _terminated, _truncated, info = env.step(action)
            next_state, reward, done, crm_info = rm.step(state, next_env_state, info)
            agent.update(action, crm_info)
            total_reward += reward
            length += 1
            state = next_state

        td_error = np.sum(agent.td_errors[-length:]) / length
        td_errors.append(td_error)
        total_rewards.append(total_reward)
        steps.append(length)

        if episode % report_each == 0:
            if verbose:
                print(
                    f"episode {episode:4}: error={td_error: .3f}, reward={total_reward:.3f}, steps={length:3.0f}, epsilon={agent.epsilon:.3f}"
                )

        agent.decay_epsilon()

    env.close()
    return td_errors, total_rewards, steps
