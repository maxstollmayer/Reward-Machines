import math
from dataclasses import dataclass

import pygame
from config import (
    ACCELERATION,
    CAR_LENGTH,
    CAR_WIDTH,
    FRICTION,
    MAX_SPEED,
    SCREEN_CENTER,
    TURNING_ANGLE,
)

Num = int | float
Vec = pygame.math.Vector2
Pos = Vec | tuple[Num, Num]


def dir(angle: Num) -> Vec:
    rad = math.radians(angle)
    return Vec((math.cos(rad), math.sin(rad)))


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


class Sprite(pygame.sprite.Sprite):
    def __init__(self, pos: Pos, angle: Num, size: Pos, sprite: str):
        super().__init__()
        self.image = pygame.image.load("src/assets/" + sprite).convert_alpha()
        self.image = pygame.transform.scale(self.image, size)
        self.surf = self.image
        self.surf = pygame.transform.rotate(self.image, angle)
        self.rect = self.surf.get_rect(center=pos)


class Entity(pygame.sprite.Sprite):
    def __init__(
        self,
        pos: Pos,
        angle: Num,
        size: Pos,
        sprite: str,
        acc: Num = 0,
        turn: Num = 0,
        max_speed: Num = 0,
    ):
        super().__init__()
        self.image = pygame.image.load("src/assets/" + sprite).convert_alpha()
        self.image = pygame.transform.scale(self.image, size)

        self.acc = acc
        self.turn = turn
        self.max_speed = max_speed
        self.vel = 0
        self.pos = Vec(pos)
        self.angle = angle

    def update(self, state: State):
        action = self.move(state)

        if action.forward and not action.backward:
            self.vel += self.acc
        if action.backward and not action.forward:
            self.vel -= self.acc / 2  # slower reversing
        if action.left:
            self.angle -= min(
                self.turn * self.vel, self.turn
            )  # no turning when standing
        if action.right:
            self.angle += min(self.turn * self.vel, self.turn)

        self.vel = max(
            min((1 - FRICTION) * self.vel, self.max_speed), -self.max_speed / 2
        )
        self.pos += self.vel * dir(self.angle)

    def draw(self, screen: pygame.Surface, camera: Vec):
        surf = pygame.transform.rotate(self.image, -self.angle)
        rect = self.pos - camera
        screen.blit(surf, rect)

    def move(self, state: State) -> Action: ...


class Player(Entity):
    def __init__(self, pos: Pos, angle: Num):
        super().__init__(
            pos,
            angle,
            (CAR_WIDTH, CAR_LENGTH),
            "car.png",
            acc=ACCELERATION,
            turn=TURNING_ANGLE,
            max_speed=MAX_SPEED,
        )
        # TODO: make car images correct angle to begin with
        #       maybe also a car that fits the theme of a car carpet
        self.image = pygame.transform.rotate(self.image, -90)

    def get_camera(self) -> Vec:
        return self.pos - SCREEN_CENTER

    def draw(self, screen: pygame.Surface, camera: Vec):
        surf = pygame.transform.rotate(self.image, -self.angle)
        rect = surf.get_rect(center=self.pos)
        rect = SCREEN_CENTER - Vec(rect.width / 2, rect.height / 2)
        screen.blit(surf, rect)

    def move(self, state: State) -> Action: ...


class Human(Player):
    def __init__(self, pos: Pos, angle: Num):
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


class Pothole(Entity):
    def __init__(self, pos: Pos):
        # TODO: create image
        super().__init__(pos, 0, (20, 20), "car.png")

    def move(self, state: State) -> Action:
        return Action()


class Crosswalker(Entity):
    def __init__(self, pos: Pos, angle: Num):
        # TODO: create image
        super().__init__(pos, angle, (CAR_WIDTH, CAR_LENGTH), "car.png")

    def move(self, state: State) -> Action:
        # TODO: implement crosswalking
        return Action()


class Car(Entity):
    def __init__(self, path: list[tuple[Num, Num]]):
        if not path:
            raise ValueError("ERROR: Path cannot be empty.")

        self.path = [Vec(v) for v in path]
        if len(self.path) == 1:
            self.target = -1
            angle = 0
        else:
            self.target = 1
            diff = self.path[self.target] - self.path[0]
            angle = -diff.angle_to((1, 0))

        super().__init__(
            self.path[0],
            angle,
            (CAR_WIDTH, CAR_LENGTH),
            "car.png",
            acc=ACCELERATION,
            turn=TURNING_ANGLE,
            max_speed=MAX_SPEED,
        )
        # TODO: make car.png in different color and correct angle
        self.image = pygame.transform.rotate(self.image, -90)

    def move(self, state: State) -> Action:
        if self.target == -1:
            return Action()

        # check if at target
        diff = self.path[self.target % len(self.path)] - self.pos
        dist = diff.length_squared()
        if dist <= CAR_LENGTH**2:
            self.target += 1
            diff = self.path[self.target % len(self.path)] - self.pos
            dist = diff.length_squared()

        # steer to next
        target_angle = -dir(self.angle).angle_to(diff)
        if target_angle > self.turn:
            return Action(forward=True, left=True)
        if target_angle < -self.turn:
            return Action(forward=True, right=True)
        return Action(forward=True)
