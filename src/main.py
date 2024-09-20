import sys

import pygame

from models import Car, Human, Pothole, State
from config import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    TITLE_CAPTION,
    FPS,
    BLACK,
    WHITE,
)


pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(TITLE_CAPTION)
clock = pygame.time.Clock()

player = Human((0, 0), 0)
# TODO: create obstacles: pothole? crosswalk? cars? cyclists?
# generate random potholes on the street at game start
# generate random crosswalkers
# generate random cars from a few different pre defined paths at start of game
# make sure not to spawn them on top of the player
# TODO: what about collisions between them?
pothole = Pothole((0, 0))
car = Car([(0, 0), (2000, 0), (2000, 1500)])


def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # TODO: calculate collision, when exactly?
        state = State(pygame.key.get_pressed(), screen)

        pothole.update(state)
        car.update(state)

        player.update(state)
        camera = player.get_camera()

        screen.fill(BLACK)
        # TODO: draw cars carpet
        pygame.draw.rect(
            screen, WHITE, (-camera[0], -camera[1], WORLD_WIDTH, WORLD_HEIGHT)
        )

        pothole.draw(screen, camera)
        car.draw(screen, camera)
        player.draw(screen, camera)

        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
