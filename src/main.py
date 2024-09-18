import sys

import pygame

from models import Human, State
from config import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SCREEN_CENTER,
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
# TODO: create obstacles: pothole? crosswalk? cars?


def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        state = State(pygame.key.get_pressed(), pygame.display.get_surface())
        player.update(state)
        camera = player.pos - SCREEN_CENTER

        # clamp camera?
        # camera[0] = max(0, min(camera[0], WORLD_WIDTH - SCREEN_WIDTH))
        # camera[1] = max(0, min(camera[1], WORLD_HEIGHT - SCREEN_HEIGHT))

        # TODO: draw cars carpet
        screen.fill(BLACK)
        pygame.draw.rect(
            screen, WHITE, (-camera[0], -camera[1], WORLD_WIDTH, WORLD_HEIGHT)
        )

        screen.blit(player.surf, player.rect)

        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
