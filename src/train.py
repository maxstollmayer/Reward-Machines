from gymnasium.wrappers import HumanRendering

from agent import Agent
from env import RMMiniGridEnv
from rm import RM

IDX_TO_ACTION = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "idle",
}


def train(
    agent: Agent[tuple[int, int], int] | Agent[int, int],
    env: RMMiniGridEnv,
    rm: RM | None = None,
    episodes: int = 1,
    report_each: int = 1,
    verbose: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    errors: list[float] = []
    rewards: list[float] = []
    steps: list[int] = []

    for episode in range(episodes):
        state, _ = env.reset()
        if rm is not None:
            u = rm.reset()
            state = (state, u)
        total_reward = 0
        total_error = 0
        length = 0

        terminal = False
        while not terminal:
            action = agent.get_action(state, explore=True)

            next_state, reward, terminated, truncated, props = env.step(action)
            terminal = terminated or truncated
            experience = {}
            if rm is not None:
                next_state, reward, terminal, experience = rm.step(
                    state, next_state, props
                )
            error = agent.update(
                state, action, reward, next_state, terminal, experience
            )
            state = next_state

            total_error += abs(error)
            total_reward += reward
            length += 1

        errors.append(total_error)
        rewards.append(total_reward)
        steps.append(length)

        if verbose and episode % report_each == 0:
            print(
                f"episode {episode:4}: error={total_error:6.2f}, reward={total_reward:4.1f}, steps={length:3.0f}, epsilon={agent.epsilon:.3f}"
            )

        agent.decay_epsilon()

    env.close()
    return errors, rewards, steps


def test(
    agent: Agent[tuple[int, int], int] | Agent[int, int],
    env: RMMiniGridEnv,
    rm: RM | None = None,
    render: bool = False,
    verbose: bool = False,
) -> int:
    if render:
        env = HumanRendering(env)
    state, _ = env.reset()
    if rm is not None:
        u = rm.reset()
        state = (state, u)
    steps = 0
    terminal = False

    while not terminal:
        action = agent.get_action(state, explore=False)
        next_state, _, terminated, truncated, props = env.step(action)
        terminal = terminated or truncated
        if rm is not None:
            next_state, _, terminal, _ = rm.step(state, next_state, props)
        state = next_state
        steps += 1
        if verbose:
            print(IDX_TO_ACTION[action])
        if render:
            env.render()

    env.close()
    return steps
