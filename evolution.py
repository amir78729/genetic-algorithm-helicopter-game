import copy
import random
from player import Player
import numpy as np
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        pass


    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover

            new_players = []
            count = 0
            while count < num_players:
                p = self.roulette_wheel(prev_players)
                child = copy.deepcopy(p)
                self.mutate(child)
                new_players.append(child)
                count += 1
                print(child.fitness, end=" ")

            return new_players

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects
        players = sorted(players, key=lambda x: x.fitness)[::-1]

        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting

        return players[: num_players]

    def roulette_wheel(self, chromosomes):
        max = sum([c.fitness for c in chromosomes])
        pick = random.uniform(0, max)
        current = 0
        for chromosome in chromosomes:
            current += chromosome.fitness
            if current > pick:
                return chromosome
