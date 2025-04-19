from agent import Agent
from env import Env
from envs.doorkey import DoorKey
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
    agent: Agent,
    env: DoorKey,
    rm: RM | None = None,
    episodes: int = 1,
    report_each: int = 1,
    verbose: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    errors: list[float] = []
    rewards: list[float] = []
    steps: list[int] = []

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        u = 0
        if rm is not None:
            u = rm.reset()
        state = (obs, u)
        total_reward = 0
        total_error = 0
        length = 0

        terminal = False
        while not terminal:
            action = agent.get_action(state, explore=True)

            next_obs, reward, terminated, truncated, props = env.step(action)
            next_state = (next_obs, 0)
            terminal = terminated or truncated
            experiences = []
            if rm is not None:
                next_state, reward, terminal, experiences = rm.step(
                    state, next_obs, props
                )
            error = agent.update(
                state, action, reward, next_state, terminal, experiences
            )
            state = next_state

            total_error += abs(error)
            total_reward += reward
            length += 1

        errors.append(total_error)
        rewards.append(total_reward)
        steps.append(length)

        if verbose and episode % report_each == 0:
            err = sum(errors[episode - report_each : episode]) / report_each
            rew = sum(rewards[episode - report_each : episode]) / report_each
            stp = sum(steps[episode - report_each : episode]) / report_each
            print(
                f"episode {episode:4}: error={err:6.2f}, reward={rew:4.1f}, steps={stp:3.0f}, epsilon={agent.epsilon:.3f}"
            )

        agent.decay_epsilon()

    env.close()
    return errors, rewards, steps


def test(
    agent: Agent,
    env: Env,
    rm: RM | None = None,
    render: bool = False,
    verbose: bool = False,
    seed: int | None = None,
) -> int:
    obs, _ = env.reset(seed=seed)
    state = (obs, 0)
    if rm is not None:
        u = rm.reset()
        state = (obs, u)
    steps = 0
    terminal = False

    while not terminal:
        action = agent.get_action(state, explore=False)
        next_obs, _, terminated, truncated, props = env.step(action)
        next_state = (next_obs, 0)
        terminal = terminated or truncated
        if rm is not None:
            next_state, _, terminal, _ = rm.step(state, next_obs, props)
        state = next_state
        steps += 1
        if verbose:
            print(IDX_TO_ACTION[action])
        if render:
            env.render()

    env.close()
    return steps
