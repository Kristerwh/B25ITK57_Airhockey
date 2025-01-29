# import pygame
# import sys
# from constants import *
# from controls import *
#
# pygame.init()
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Airhockey simu")
# logo = pygame.image.load("hiof.png")
# pygame.display.set_icon(logo)
# clock = pygame.time.Clock()
#
# player1 = [40, HEIGHT // 2]
# player2 = [WIDTH - 40, HEIGHT // 2]
# puck_position = [WIDTH // 2, HEIGHT // 2]
# puck_speed = [5, 5]
#
# def draw_table():
#     screen.fill(COLORS["black"])
#     pygame.draw.line(screen, COLORS["white"], (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 5)
#     pygame.draw.circle(screen, COLORS["white"], (WIDTH // 2, HEIGHT // 2), 50, 5)
#
# def draw_game_elements():
#     pygame.draw.circle(screen, COLORS["red"], player1, 30)
#     pygame.draw.circle(screen, COLORS["blue"], player2, 30)
#     pygame.draw.circle(screen, COLORS["white"], puck_position, 20)
#
# def update_puck():
#     puck_position[0] -= abs(puck_speed[0])
#     puck_position[1] = HEIGHT // 2
#
#
#     if puck_position[0] - 20 <= 0:
#         puck_position[0] = WIDTH // 2
#     if puck_position[1] - 20 <= 0:
#         puck_position[1] = HEIGHT // 2