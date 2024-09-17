from abc import ABC
from dataclasses import dataclass
import math
from random import choice
import sys
from typing import Any

import pygame

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
TILE_SIZE = 32

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)

TITLE_CAPTION = "test game"  # TODO: change

CAR_WIDTH = 30
CAR_LENGTH = 50
ACCELERATION = 0.2
TURNING_ANGLE = math.radians(3)
FRICTION = 0.05
MAX_SPEED = 8

num = int | float
vec = pygame.math.Vector2
pos = vec | tuple[num, num]


pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(TITLE_CAPTION)
clock = pygame.time.Clock()


@dataclass
class Action:
    forward: bool = False
    backward: bool = False
    left: bool = False
    right: bool = False

    @classmethod
    def number(cls) -> int:
        return 4


@dataclass
class State:
    pressed_keys: Any
    screen: Any


class Car(pygame.sprite.Sprite, ABC):
    def __init__(self, pos: pos, angle: num, sprite: str):
        super().__init__()
        self.image = pygame.image.load(sprite).convert_alpha()
        self.image = pygame.transform.rotate(self.image, -90)
        self.image = pygame.transform.scale(self.image, (CAR_LENGTH, CAR_WIDTH))
        self.surf = self.image
        self.surf = pygame.transform.rotate(self.image, angle)
        self.rect = self.surf.get_rect(center=pos)

        self.pos = vec(pos)
        self.vel = 0
        self.angle = math.radians(angle)

    def move(self, state: State) -> Action:
        raise NotImplementedError()

    def update(self, state: State):
        action = self.move(state)

        if action.forward and not action.backward:
            self.vel += ACCELERATION
        if action.backward and not action.forward:
            self.vel -= ACCELERATION / 2  # slower reversing
        if action.left:
            self.angle -= min(TURNING_ANGLE * self.vel, TURNING_ANGLE)
        if action.right:
            self.angle += min(TURNING_ANGLE * self.vel, TURNING_ANGLE)

        dir = vec((math.cos(self.angle), math.sin(self.angle)))
        self.vel = max(min((1 - FRICTION) * self.vel, MAX_SPEED), -MAX_SPEED / 2)
        self.pos += self.vel * dir

        if self.pos.x > SCREEN_WIDTH:
            self.pos.x = 0
        if self.pos.x < 0:
            self.pos.x = SCREEN_WIDTH
        if self.pos.y > SCREEN_HEIGHT:
            self.pos.y = 0
        if self.pos.y < 0:
            self.pos.y = SCREEN_HEIGHT

        # self.rect.center = self.pos  # type: ignore

        self.surf = pygame.transform.rotate(self.image, -math.degrees(self.angle))
        self.rect = self.surf.get_rect(center=self.pos)


class Player(Car):
    def __init__(self, pos: pos, angle: num):
        super().__init__(pos, angle, "car.png")

    def move(self, state: State) -> Action:
        action = Action()
        if state.pressed_keys[pygame.K_UP]:
            action.forward = True
        if state.pressed_keys[pygame.K_DOWN]:
            action.backward = True
        if state.pressed_keys[pygame.K_LEFT]:
            action.left = True
        if state.pressed_keys[pygame.K_RIGHT]:
            action.right = True
        return action


class Agent(Car):
    def __init__(self, pos: pos, angle: num):
        super().__init__(pos, angle, "car.png")

    def move(self, state: State) -> Action:
        # TODO: implement RL
        forward = choice([True, False])
        left = choice([True, False])
        return Action(forward, not forward, left, not left)


class Obstacle(Car):
    def __init__(self, path: list[tuple[num, num]]):
        self.path = [vec(v) for v in path[::-1]]
        start = self.path.pop()
        self.target = self.path.pop()
        diff = self.target - start
        angle = -diff.angle_to((1, 0))
        super().__init__(start, angle, "car.png")
        # TODO: make car.png in different color
        self.stop = False

    def move(self, state: State) -> Action:
        if self.stop:
            return Action()

        # check if at target
        diff = self.target - self.pos
        dist = diff.length_squared()
        if dist <= 100:
            if not self.path:
                self.stop = True
                return Action()
            self.target = self.path.pop()

        # steer to next
        dir = vec(math.cos(self.angle), math.sin(self.angle))
        target_angle = -dir.angle_to(diff)
        forward = True  # dist > CAR_LENGTH**2
        if target_angle > 5:
            return Action(forward=forward, left=True)
        if target_angle < -5:
            return Action(forward=forward, right=True)
        return Action(forward=forward)


player = Player((100, 200), 0)
ai = Agent((SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2), 0)
obstacle = Obstacle([(100, 100), (200, 500), (400, 200), (700, 500)])
cars = [player, ai, obstacle]


def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for car in cars:
            car.update(State(pygame.key.get_pressed(), None))
        # player.update(State(pygame.key.get_pressed(), None))
        # ai.update(State(None, None))
        # TODO: get field of view of the ai
        # or maybe just the whole screen as a starting point
        # how to handle edges? different color of pixels?
        # obstacle.update(State(None, None))

        # TODO: implement rewards
        # TODO: implement collision for negative rewards

        screen.fill(BLACK)

        for car in cars:
            screen.blit(car.surf, car.rect)
        # screen.blit(player.surf, player.rect)
        # screen.blit(ai.surf, ai.rect)
        # screen.blit(obstacle.surf, obstacle.rect)

        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
