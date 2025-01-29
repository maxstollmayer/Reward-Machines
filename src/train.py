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


def train_RMLearner(
    agent: RMLearner,
    env: RMMiniGridEnv,
    rm: RM,
    episodes: int = 1000,
    report_each: int = 100,
    verbose: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    errors: list[float] = []
    rewards: list[float] = []
    steps: list[int] = []
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

        error = sum(agent.errors[-length:]) / length
        errors.append(error)
        rewards.append(total_reward)
        steps.append(length)

        if episode % report_each == 0:
            if verbose:
                print(
                    f"episode {episode:4}: error={error: .3f}, reward={total_reward:.3f}, steps={length:3.0f}, epsilon={agent.epsilon:.3f}"
                )

        agent.decay_epsilon()

    env.close()
    return errors, rewards, steps


def test_RMLearner(
    agent: RMLearner,
    env: RMMiniGridEnv,
    rm: RM,
    render: bool = False,
    verbose: bool = False,
) -> int:
    if render:
        env = HumanRendering(env)
    env_state, _ = env.reset(seed=0)
    u = rm.reset()
    state = (env_state, u)
    done = False
    steps = 0

    while not done:
        action = agent.get_action(state, explore=False)
        next_env_state, _, _, _, info = env.step(action)
        state, _, done, _ = rm.step(state, next_env_state, info)
        steps += 1
        if verbose:
            print(IDX_TO_ACTION[action])
        if render:
            env.render()

    env.close()
    return steps


def train_QLearner(
    agent: QLearner,
    env: RMMiniGridEnv,
    episodes: int = 1000,
    report_each: int = 100,
    verbose: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    errors: list[float] = []
    rewards: list[float] = []
    steps: list[int] = []
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

        error = sum(agent.errors[-length:]) / length
        errors.append(error)
        rewards.append(total_reward)
        steps.append(length)

        if episode % report_each == 0:
            if verbose:
                print(
                    f"episode {episode:4}: error={error: .3f}, reward={total_reward:.3f}, steps={length:3.0f}, epsilon={agent.epsilon:.3f}"
                )

        agent.decay_epsilon()

    env.close()
    return errors, rewards, steps


def test_QLearner(
    agent: QLearner, env: RMMiniGridEnv, render: bool = False, verbose: bool = False
) -> int:
    if render:
        env = HumanRendering(env)
    state, _ = env.reset(seed=0)
    steps = 0
    done = False

    while not done:
        action = agent.get_action(state, explore=False)
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated
        if verbose:
            print(IDX_TO_ACTION[action])
        if render:
            env.render()

    env.close()
    return steps
