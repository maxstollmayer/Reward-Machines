from enum import Enum, auto

type State = int
type Reward = float

class Action(Enum):
    Up = auto()
    Down = auto()
    Left = auto()
    Right = auto()


class RandomAgent:
    def act(self, state: State) -> Action:
        return Action.Up

    def update(self, state: State, action: Action, reward: Reward, done: bool):
        pass


class GridWorld:
    def __init__(self):
        # give grid as input: list[list[Symbol]]
        # what is a state? just a coordinate or does it encode more info like
        # the whole grid with the actor moved
        self._grid = [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self._width = 5
        self._height = 4
        self._state = 0
        self._goal = 19

    def get_index(self) -> tuple[int, int]:
        y = self._state % self._height
        x = self._state // self._height
        return (x, y)


    def reset(self) -> State:
        self._state = 0
        return 0

    def step(self, action: Action) -> tuple[State, Reward, bool]:
        x, y = self._state
        return (x, y)


def main():
    print("Hello World!")


if __name__ == "__main__":
    main()
