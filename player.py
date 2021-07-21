import math
import pygame
import numpy as np
import random

from nn import NeuralNetwork
from config import CONFIG


class Player():

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

        if mode in ['helicopter', 'gravity']:
            if len(box_lists):
                target_position = [box_lists[0].x, box_lists[0].gap_mid]
            else:
                target_position = [CONFIG['WIDTH'], CONFIG['HEIGHT'] / 2]
            dist = 120
            dist_up = CONFIG['HEIGHT'] - agent_position[1]
            dist_down = agent_position[1]
            up_gap = abs(agent_position[1] - dist - target_position[1])
            down_gap = abs(agent_position[1] + dist - target_position[1])
            d_distance = math.sqrt(((agent_position[0] - target_position[0]) ** 2 + (agent_position[1] - target_position[1]) ** 2))

            vector_max = max([dist_up, up_gap, d_distance, down_gap, dist_down, 100 * velocity])
            vector = [dist_up / vector_max, up_gap / vector_max, d_distance / vector_max, down_gap / vector_max,
                      dist_down / vector_max, 100 * velocity / vector_max]
            self.nn.forward(vector)
            if self.nn.output_layer[0][0] > 0.5:
                direction = 1
            else:
                direction = -1
            return direction
            stateVector = np.zeros((6, 1))

            max_dist_x_position = 850
            max_y_position = CONFIG['HEIGHT']
            max_velocity = 6
            if mode == 'thrust':
                max_velocity = 500
            max_dist_boxes_position = 500
            max_gap_mid = 500

            if len(box_lists) > 1:
                stateVector[0][0] = (agent_position[0] - box_lists[0].x) / max_dist_x_position
                stateVector[1][0] = agent_position[1] / max_y_position
                stateVector[2][0] = velocity / max_velocity
                stateVector[3][0] = (box_lists[1].x - box_lists[0].x) / max_dist_boxes_position
                stateVector[4][0] = box_lists[0].gap_mid / max_gap_mid
                stateVector[5][0] = box_lists[1].gap_mid / max_gap_mid

            direction = -1
            if len(box_lists) > 1:
                self.nn.forward(stateVector)
                if mode == 'helicopter':
                    if self.nn.output_layer[0][0] > 0.5:
                        direction = 1
                if mode == 'gravity':
                    if self.nn.output_layer[0][0] > 0.5:
                        direction = 1
                if mode == 'thrust':
                    if self.nn.output_layer[0][0] > 0.5:
                        direction = 1
                    # if yhat <= 0.33:
                    #     direction = -1
                    # elif yhat > 0.33 and yhat <= 0.66:
                    #     direction = 0
                    # else:
                    #     direction = 1
            else:
                randomNumber = random.random()
                if randomNumber > 0.5:
                    direction = 1

            return direction

        if mode == 'thrust':
            if len(box_lists):
                target_position = [box_lists[0].x, box_lists[0].gap_mid]
            else:
                target_position = [CONFIG['WIDTH'], CONFIG['HEIGHT'] / 2]
            dist = 120
            dist_up = CONFIG['HEIGHT'] - agent_position[1]
            dist_down = agent_position[1]
            up_gap = abs(agent_position[1] - dist - target_position[1])
            down_gap = abs(agent_position[1] + dist - target_position[1])
            d_distance = math.sqrt(((agent_position[0] - target_position[0]) ** 2 + (agent_position[1] - target_position[1]) ** 2))

            vector_max = max([dist_up, up_gap, d_distance, down_gap, dist_down, 100 * velocity])
            vector = [dist_up / vector_max, up_gap / vector_max, d_distance / vector_max, down_gap / vector_max,
                      dist_down / vector_max, 100 * velocity / vector_max]
            self.nn.forward(vector)
            if self.nn.output_layer[0][0] < 0.3:
                direction = -1
            elif self.nn.output_layer[0][0] < 0.7:
                direction = 0
            else:
                direction = 1
            return direction



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
