from random import random, choice
from dataclasses import dataclass
import pygame
import math

from config import (
    SCREEN_CENTER,
    CAR_LENGTH,
    CAR_WIDTH,
    ACCELERATION,
    FRICTION,
    TURNING_ANGLE,
    MAX_SPEED,
)

num = int | float
vec = pygame.math.Vector2
pos = vec | tuple[num, num]


def dir_vec(angle: num) -> vec:
    return vec((math.cos(angle), math.sin(angle)))


@dataclass
class Action:
    forward: bool = False
    backward: bool = False
    left: bool = False
    right: bool = False


@dataclass
class State:
    keys: pygame.key.ScancodeWrapper
    screen: pygame.Surface


class Player(pygame.sprite.Sprite):
    def __init__(self, pos: pos, angle: num):
        super().__init__()
        self.image = pygame.image.load("car.png").convert_alpha()
        self.image = pygame.transform.rotate(self.image, -90)
        # TODO: make car images correct angle to begin with
        # maybe also a car that fits the theme of a car carpet
        self.image = pygame.transform.scale(self.image, (CAR_LENGTH, CAR_WIDTH))
        self.surf = self.image
        self.surf = pygame.transform.rotate(self.image, angle)
        self.rect = self.surf.get_rect(center=pos)

        self.pos = vec(pos)
        self.vel = 0
        self.angle = math.radians(angle)
        self.dir = dir_vec(self.angle)

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

        self.vel = max(min((1 - FRICTION) * self.vel, MAX_SPEED), -MAX_SPEED / 2)
        self.pos += self.vel * dir_vec(self.angle)

        self.surf = pygame.transform.rotate(self.image, -math.degrees(self.angle))
        self.rect = self.surf.get_rect(center=self.pos)
        self.rect = SCREEN_CENTER - vec(self.rect.width / 2, self.rect.height / 2)

    def move(self, state: State) -> Action: ...


class Human(Player):
    def __init__(self, pos: pos, angle: num):
        super().__init__(pos, angle)

    def move(self, state: State) -> Action:
        action = Action()
        if state.keys[pygame.K_UP]:
            action.forward = True
        if state.keys[pygame.K_DOWN]:
            action.backward = True
        if state.keys[pygame.K_LEFT]:
            action.left = True
        if state.keys[pygame.K_RIGHT]:
            action.right = True
        return action


class Agent(Player):
    def __init__(self, pos: pos, angle: num):
        super().__init__(pos, angle)

    def move(self, state: State) -> Action:
        # TODO: implement RL
        forward = random() > 0.25
        left = choice([True, False])
        return Action(forward, False, left, not left)
