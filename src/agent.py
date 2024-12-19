from dataclasses import dataclass, field

import numpy as np

# TODO: use defaultdicts instead of handrolled ones below:


@dataclass
class RMLearner:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.1
    weights: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    td_errors: list[float] = field(default_factory=list)

    def get_row(self, state: tuple[int, int]) -> list[float]:
        row = self.weights.get(state)
        if row is None:
            row = list(np.random.rand(self.n_actions))
            self.weights[state] = row
        return row

    def get_q(self, state: tuple[int, int], action: int) -> float:
        row = self.weights.get(state)
        if row is None:
            row = list(np.random.rand(self.n_actions))
            self.weights[state] = row
        return row[action]

    def set_q(self, state: tuple[int, int], action: int, q: float) -> None:
        row = self.weights.get(state)
        if row is None:
            row = list(np.random.rand(self.n_actions))
            self.weights[state] = row
        row[action] = q

    def explore(self) -> int:
        return np.random.randint(self.n_actions)

    def exploit(self, state: tuple[int, int]) -> int:
        return int(np.argmax(self.get_row(state)))

    def get_action(self, state: tuple[int, int], explore: bool = False) -> int:
        # espilon greedy action selection
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
        q = self.get_q(state, action)
        future_q: float = (not terminal) * np.max(self.get_row(next_state))
        td_target = reward + self.gamma * future_q
        new_q = (1 - self.alpha) * q + self.alpha * td_target
        self.set_q(state, action, new_q)
        self.td_errors.append(td_target - q)

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
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.1
    weights: dict[int, list[float]] = field(default_factory=dict)
    td_errors: list[float] = field(default_factory=list)

    def get_row(self, state: int) -> list[float]:
        row = self.weights.get(state)
        if row is None:
            row = list(np.random.rand(self.n_actions))
            self.weights[state] = row
        return row

    def get_q(self, state: int, action: int) -> float:
        row = self.weights.get(state)
        if row is None:
            row = list(np.random.rand(self.n_actions))
            self.weights[state] = row
        return row[action]

    def set_q(self, state: int, action: int, q: float) -> None:
        row = self.weights.get(state)
        if row is None:
            row = list(np.random.rand(self.n_actions))
            self.weights[state] = row
        row[action] = q

    def explore(self) -> int:
        return np.random.randint(self.n_actions)

    def exploit(self, state: int) -> int:
        return int(np.argmax(self.get_row(state)))

    def get_action(self, state: int, explore: bool = False) -> int:
        # espilon greedy action selection
        if explore and np.random.rand() < self.epsilon:
            return self.explore()
        return self.exploit(state)

    def update(
        self, state: int, action: int, reward: float, terminal: bool, next_state: int
    ) -> None:
        # update q value using Bellman equation
        q = self.get_q(state, action)
        # TODO: why better performance without `not terminal`?
        future_q: float = (not terminal) * np.max(self.get_row(next_state))
        td_target = reward + self.gamma * future_q
        new_q = (1 - self.alpha) * q + self.alpha * td_target
        self.set_q(state, action, new_q)
        self.td_errors.append(td_target - q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
