from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, override

import numpy as np


class Agent[State, Action](Protocol):
    def get_action(self, state: State, explore: bool) -> Action: ...
    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        terminal: bool,
        experience: dict[State, tuple[State, float, bool]],
    ) -> float: ...
    def decay_epsilon(self) -> None: ...


@dataclass
class QAgent[State]:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    weights: defaultdict[State, list[float]] = field(init=False)

    def __post_init__(self) -> None:
        def default_factory() -> list[float]:
            return [np.random.rand() for _ in range(self.n_actions)]

        self.weights = defaultdict(default_factory)

    def get_action(self, state: State, explore: bool = False) -> int:
        # epsilon greedy action selection
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.weights[state]))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        terminal: bool,
        experience: dict[State, tuple[State, float, bool]],
    ) -> float:
        # Bellman update
        current_q = self.weights[state][action]
        future_q: float = np.max(self.weights[next_state]) * (not terminal)
        td_error = reward + self.gamma * future_q - current_q
        self.weights[state][action] = current_q + self.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


QLearner = QAgent[int]
BLLearner = QAgent[tuple[int, int]]


class CRMLearner(QAgent[tuple[int, int]]):
    @override
    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        terminal: bool,
        experience: dict[tuple[int, int], tuple[tuple[int, int], float, bool]],
    ) -> float:
        # update q table for every rm state
        td_error = 0
        for _state, (_reward, _next_state, _terminal) in experience.items():
            error = super().update(_state, action, _next_state, _reward, _terminal, {})
            if _state == state:
                td_error = error
        return td_error


@dataclass
class DQNAgent[State]:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    hidden_dim: int = 128
    w1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # TODO: need to change environment for this to give raw array as observation not hashes
        # maybe also change to full observation instead of partial
        # for rgb array convolutional layer would be better

        state_dim = 1 # TODO: or maybe something like np.shape(State())
        self.w1 = np.random.randn(state_dim, self.hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = np.random.randn(self.hidden_dim, self.n_actions) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(self.n_actions)
    def _forward(self, state: np.ndarray) -> np.ndarray: ...
    def _backward(self, state, np.ndarray, action: int, target: float) -> None: ...
    def get_action(self, state: State, explore: bool = False) -> int: ...
    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: tuple[int, int],
        terminal: bool,
        experience: dict[tuple[int, int], tuple[tuple[int, int], float, bool]],
    ) -> float: ...
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
