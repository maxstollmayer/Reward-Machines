import sys

import pygame

from models import Player, Agent, Obstacle, State
from config import SCREEN_HEIGHT, SCREEN_WIDTH, TITLE_CAPTION, BLACK, FPS


pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(TITLE_CAPTION)
clock = pygame.time.Clock()

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

        # TODO: get field of view of the ai
        # or maybe just the whole screen as a starting point
        # how to handle edges? different color of pixels?
        # make whole screen the view of ai and the map bigger with scrolling

        state = State(pygame.key.get_pressed(), None)
        for car in cars:
            car.update(state)

        # TODO: implement rewards
        # TODO: implement collision for negative rewards
        # how to handle training vs inference?

        screen.fill(BLACK)

        for car in cars:
            screen.blit(car.surf, car.rect)

        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
