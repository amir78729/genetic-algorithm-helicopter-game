import copy
import random
from player import Player
import numpy as np
import csv
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
        mutation_prob = 0.6
        mutation_std = 0.9
        r = np.random.rand()
        self.mutate_array(child.nn.weights[0], mutation_prob, mutation_std, r)
        self.mutate_array(child.nn.biases[0], mutation_prob, mutation_std, r)
        self.mutate_array(child.nn.weights[1], mutation_prob, mutation_std, r)
        self.mutate_array(child.nn.biases[1], mutation_prob, mutation_std, r)


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
        # getting statistic data for the fitness of players
        fitnesses = np.array([x.fitness for x in players])
        max_fitness = np.max(fitnesses)
        min_fitness = np.min(fitnesses)
        avg_fitness = np.average(fitnesses)
        print('\n - max: {}\n - min: {}\n - avg: {}'.format(max_fitness, min_fitness, avg_fitness))

        # writing statistic data on a csv file
        with open(r'fitness_data_for_plotting.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([min_fitness, avg_fitness, max_fitness])

        return players[: num_players]

    def roulette_wheel(self, chromosomes):
        max = sum([c.fitness for c in chromosomes])
        pick = random.uniform(0, max)
        current = 0
        for chromosome in chromosomes:
            current += chromosome.fitness
            if current > pick:
                return chromosome

    def mutate_array(self, array, prob, std, r):
        seed = random.randint(0, 1000000000)
        rs = np.random.RandomState(seed)

        # change weights
        mutation_mask = rs.random(array.shape) < prob
        while np.sum(mutation_mask) == 0:
            mutation_mask = rs.random(array.shape) < prob
        mutation = mutation_mask * rs.normal(0, std, array.shape)
        # print("before:")
        # print(array)
        array += mutation
        # print("after")
        # print(array)