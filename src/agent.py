from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RMLearner:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    weights: defaultdict[tuple[int, int], list[float]] = field(init=False)
    errors: list[float] = field(default_factory=list)

    def __post_init__(self):
        def default_factory() -> list[float]:
            return [np.random.rand() for _ in range(self.n_actions)]

        self.weights = defaultdict(default_factory)

    def explore(self) -> int:
        return np.random.randint(self.n_actions)

    def exploit(self, state: tuple[int, int]) -> int:
        return int(np.argmax(self.weights[state]))

    def get_action(self, state: tuple[int, int], explore: bool = False) -> int:
        # epsilon greedy action selection
        if explore and np.random.rand() < self.epsilon:
            return self.explore()
        return self.exploit(state)

    def bellman_update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        terminal: bool,
    ) -> None:
        current_q = self.weights[state][action]
        future_q: float = np.max(self.weights[next_state]) * (not terminal)
        td_error = reward + self.gamma * future_q - current_q
        self.weights[state][action] = current_q + self.alpha * td_error
        self.errors.append(td_error)

    def update(
        self,
        action: int,
        info: dict[tuple[int, int], tuple[tuple[int, int], float, bool]],
    ) -> None:
        for state, (next_state, reward, terminal) in info.items():
            self.bellman_update(state, action, reward, next_state, terminal)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


@dataclass
class QLearner:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    weights: defaultdict[int, list[float]] = field(init=False)
    errors: list[float] = field(default_factory=list)

    def __post_init__(self):
        def default_factory() -> list[float]:
            return [np.random.rand() for _ in range(self.n_actions)]

        self.weights = defaultdict(default_factory)

    def explore(self) -> int:
        return np.random.randint(self.n_actions)

    def exploit(self, state: int) -> int:
        return int(np.argmax(self.weights[state]))

    def get_action(self, state: int, explore: bool = False) -> int:
        # epsilon greedy action selection
        if explore and np.random.rand() < self.epsilon:
            return self.explore()
        return self.exploit(state)

    def update(
        self, state: int, action: int, reward: float, terminal: bool, next_state: int
    ) -> None:
        # update q value using Bellman equation
        current_q = self.weights[state][action]
        future_q: float = np.max(self.weights[next_state]) * (not terminal)
        td_error = reward + self.gamma * future_q - current_q
        self.weights[state][action] = current_q + self.alpha * td_error
        self.errors.append(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
