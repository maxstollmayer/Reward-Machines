from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")
type Reward = float


class Agent[State, Action](ABC):

    @abstractmethod
    def act(self, state: State) -> Action:
        pass

    @abstractmethod
    def update(self, state: State, action: Action, reward: Reward, done: bool):
        pass


class Environment[State, Action](ABC):

    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def step(self, action: Action) -> tuple[State, Reward, bool]:
        pass

    @abstractmethod
    def render(self):
        pass
