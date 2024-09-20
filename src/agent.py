from models import Action, State, Player, Pos, Num
from random import random, choice


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
