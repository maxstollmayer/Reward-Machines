from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, override

import numpy as np

from utils import Action, Experiences, Mat, Reward, State, Vec, hash_state


class Agent(Protocol):
    epsilon: float

    def get_action(self, state: State, explore: bool) -> Action: ...
    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        terminal: bool,
        experiences: Experiences,
    ) -> float: ...
    def decay_epsilon(self) -> None: ...


@dataclass
class QAgent:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    weights: defaultdict[tuple[int, ...], Vec] = field(init=False)

    def __post_init__(self) -> None:
        self.weights = defaultdict(lambda: np.random.rand(self.n_actions))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_action(self, state: State, explore: bool = False) -> Action:
        # epsilon greedy action selection
        key = hash_state(state)
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.weights[key]))

    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        terminal: bool,
        experiences: Experiences,
    ) -> float:
        # Bellman update
        k1 = hash_state(state)
        k2 = hash_state(next_state)
        current_q = self.weights[k1][action]
        future_q = (not terminal) * self.gamma * np.max(self.weights[k2])
        td_error = reward + future_q - current_q
        self.weights[k1][action] = current_q + self.alpha * td_error
        return td_error


class CRMQAgent(QAgent):
    @override
    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        terminal: bool,
        experiences: Experiences,
    ) -> float:
        # update q table for every rm state
        td_error = 0
        for _state, _reward, _next_state, _terminal in experiences:
            error = super().update(_state, action, _next_state, _reward, _terminal, [])
            if _state == state:
                td_error = error
        return td_error


@dataclass
class DQNAgent:
    n_actions: int
    state_dim: int
    hidden_dim: int = 256
    alpha: float = 1e-4
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    w1: Mat = field(init=False)
    b1: Vec = field(init=False)
    w2: Mat = field(init=False)
    b2: Vec = field(init=False)

    def __post_init__(self) -> None:
        self.w1 = np.random.randn(self.state_dim, self.hidden_dim) * np.sqrt(
            2.0 / self.state_dim
        )
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = np.random.randn(self.hidden_dim, self.n_actions) * np.sqrt(
            2.0 / self.hidden_dim
        )
        self.b2 = np.zeros(self.n_actions)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _forward(self, state: Vec) -> Vec:
        # h = np.tanh(state @ self.w1 + self.b1)
        h = np.maximum(0, state @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def _backward(self, state: Vec, action: Action, target: float) -> float:
        # h = np.tanh(state @ self.w1 + self.b1)
        h_pre = state @ self.w1 + self.b1
        h = np.maximum(0, h_pre)
        q = h @ self.w2 + self.b2
        q_pred = q[action]
        error = q_pred - target

        dL_dq = np.zeros_like(q)
        dL_dq[action] = 2 * error

        dL_dw2 = np.outer(h, dL_dq)
        dL_db2 = dL_dq

        # dh = (dL_dq @ self.w2.T) * (1 - h**2)
        dh = (dL_dq @ self.w2.T) * (h_pre > 0).astype(float)
        dL_dw1 = np.outer(state, dh)
        dL_db1 = dh

        self.w2 -= self.alpha * dL_dw2
        self.b2 -= self.alpha * dL_db2
        self.w1 -= self.alpha * dL_dw1
        self.b1 -= self.alpha * dL_db1

        return -float(error)

    def get_action(self, state: State, explore: bool = False) -> int:
        state_vec = np.array(np.append(*state), dtype=np.float64)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self._forward(state_vec)
        return int(np.argmax(q))

    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        terminal: bool,
        experiences: Experiences,
    ) -> float:
        state_vec = np.array(np.append(*state), dtype=np.float64)
        next_vec = np.array(np.append(*next_state), dtype=np.float64)
        q_next = self._forward(next_vec)
        target = reward + self.gamma * np.max(q_next) * (not terminal)
        error = self._backward(state_vec, action, target)
        return error


class CRMDQNAgent(DQNAgent):
    @override
    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        terminal: bool,
        experiences: Experiences,
    ) -> float:
        # update q table for every rm state
        td_error = 0
        for _state, _reward, _next_state, _terminal in experiences:
            error = super().update(_state, action, _next_state, _reward, _terminal, [])
            if _state == state:
                td_error = error
        return td_error
