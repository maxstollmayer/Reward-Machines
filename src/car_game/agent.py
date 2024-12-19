from dataclasses import dataclass, field
from random import choice, random

import numpy as np

from models import Action, Num, Player, Pos, State


class Agent(Player):
    def __init__(self, pos: Pos, angle: Num):
        super().__init__(pos, angle)

    def move(self, state: State) -> Action:
        # TODO: implement RL
        # what is the goal, when should it end?
        # after a crash (like out of street, into obstacle) and after a successful loop?
        forward = random() > 0.25
        left = choice([True, False])
        return Action(forward, False, left, not left)


@dataclass
class QLearner:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    weights: dict[int, list[float]] = field(default_factory=dict)

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

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        # update q value using Bellman equation
        best_next_action = self.exploit(next_state)
        td_target = reward + self.gamma * self.get_q(next_state, best_next_action)
        q = self.get_q(state, action)
        td_error = td_target - q
        self.set_q(state, action, q + self.alpha * td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
