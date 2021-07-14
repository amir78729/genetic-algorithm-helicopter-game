import pygame
import numpy as np
import math

from nn import NeuralNetwork
from config import CONFIG


class Player:

    def __init__(self, mode, control=False):

        self.control = control  # if True, playing mode is activated. else, AI mode.
        self.pos = [100, 275]   # position of the agent
        self.direction = -1     # if 1, goes upwards. else, goes downwards.
        self.v = 0              # vertical velocity
        self.g = 9.8            # gravity constant
        self.mode = mode        # game mode

        # neural network architecture (AI mode)
        layer_sizes = self.init_network(mode)

        self.nn = NeuralNetwork(layer_sizes)
        self.fitness = 0  # fitness of agent

    def move(self, box_lists, camera, events=None):

        if len(box_lists) != 0:
            if box_lists[0].x - camera + 60 < self.pos[0]:
                box_lists.pop(0)

        mode = self.mode

        # manual control
        if self.control:
            self.get_keyboard_input(mode, events)

        # AI control
        else:
            agent_position = [camera + self.pos[0], self.pos[1]]
            self.direction = self.think(mode, box_lists, agent_position, self.v)

        # game physics
        if mode == 'gravity' or mode == 'helicopter':
            self.v -= self.g * self.direction * (1 / 60)
            self.pos[1] += self.v

        elif mode == 'thrust':
            self.v -= 6 * self.direction
            self.pos[1] += self.v * (1 / 40)

        # collision detection
        is_collided = self.collision_detection(mode, box_lists, camera)

        return is_collided

    # reset agent parameters
    def reset_values(self):
        self.pos = [100, 275]
        self.direction = -1
        self.v = 0

    def get_keyboard_input(self, mode, events=None):

        if events is None:
            events = pygame.event.get()

        if mode == 'helicopter':
            self.direction = -1
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.direction = 1

        elif mode == 'thrust':
            self.direction = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.direction = 1
            elif keys[pygame.K_DOWN]:
                self.direction = -1

        for event in events:
            if event.type == pygame.KEYDOWN:

                if mode == 'gravity' and event.key == pygame.K_SPACE:
                    self.direction *= -1

    def init_network(self, mode):

        # you can change the parameters below

        layer_sizes = None
        if mode == 'gravity':
            layer_sizes = [6, 20, 1]
        elif mode == 'helicopter':
            layer_sizes = [6, 20, 1]
        elif mode == 'thrust':
            layer_sizes = [6, 20, 1]
        return layer_sizes

    def think(self, mode, box_lists, agent_position, velocity):

        # TODO
        # mode example: 'helicopter'
        # box_lists: an array of `BoxList` objects
        # agent_position example: [600, 250]
        # velocity example: 7
        # if mode == 'helicopter':
        #     mode_num = 1
        # elif mode == 'thrust':
        #     mode_num = 2
        # else:
        #     mode_num = 3

        if len(box_lists):  # there is at least one box in screen
            target_position = np.array([box_lists[0].x, box_lists[0].gap_mid])
        else:  # there is no box in the screen
            target_position = np.array([CONFIG['WIDTH'], CONFIG['HEIGHT'] / 2])
        gap = 120

        distance_from_top = CONFIG['HEIGHT'] - agent_position[1]
        distance_from_bottom = agent_position[1]

        distance_from_boxes_top = abs(agent_position[1] - target_position[1] - gap)
        distance_from_boxes_bottom = abs(agent_position[1] - target_position[1] + gap)

        diagonal_distanse = np.linalg.norm(agent_position - target_position)
        # diagonal_distanse = math.sqrt((agent_position[0] - target_position[0])**2 +
        #                               (agent_position[1] - target_position[1])**2)

        nn_input = np.array([[velocity],
                             [diagonal_distanse],
                             [distance_from_top],
                             [distance_from_bottom],
                             [distance_from_boxes_top],
                             [distance_from_boxes_bottom]])

        # normalizing input
        nn_input = nn_input / np.max(nn_input)

        # calculating output
        res = self.nn.forward(nn_input)  # 0 < res < 1

        # finding direction
        res -= .5  # -0.5 < res < 0.5
        direction = res / np.abs(res)  # direction = [[1.]] or [[-1.]]
        return int(direction[0])

    def collision_detection(self, mode, box_lists, camera):
        if mode == 'helicopter':
            rect = pygame.Rect(self.pos[0], self.pos[1], 100, 50)
        elif mode == 'gravity':
            rect = pygame.Rect(self.pos[0], self.pos[1], 70, 70)
        elif mode == 'thrust':
            rect = pygame.Rect(self.pos[0], self.pos[1], 110, 70)
        else:
            rect = pygame.Rect(self.pos[0], self.pos[1], 50, 50)
        is_collided = False

        if self.pos[1] < -60 or self.pos[1] > CONFIG['HEIGHT']:
            is_collided = True

        if len(box_lists) != 0:
            box_list = box_lists[0]
            for box in box_list.boxes:
                box_rect = pygame.Rect(box[0] - camera, box[1], 60, 60)
                if box_rect.colliderect(rect):
                    is_collided = True

        return is_collided
