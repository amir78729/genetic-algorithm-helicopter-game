from player import Player
import numpy as np
from config import CONFIG
import random
import copy
import csv
import math


class Evolution:

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate_array(self, array, prob, std, r):
        seed = random.randint(0, 1000000000)
        rs = np.random.RandomState(seed)

        # change weights
        mutation_mask = rs.random(array.shape) < prob
        while np.sum(mutation_mask) == 0:
            mutation_mask = rs.random(array.shape) < prob
        mutation = mutation_mask * rs.normal(0, std, array.shape)
        array += mutation

    def mutate(self, child):
        mutation_prob = 0.6
        mutation_std = 0.9
        r = np.random.rand()
        self.mutate_array(child.nn.w0, mutation_prob, mutation_std, r)
        self.mutate_array(child.nn.b0, mutation_prob, mutation_std, r)
        self.mutate_array(child.nn.w1, mutation_prob, mutation_std, r)
        self.mutate_array(child.nn.b1, mutation_prob, mutation_std, r)

    def roulette_wheel(self, chromosomes):
        max = sum([c.fitness for c in chromosomes])
        pick = random.uniform(0, max)
        current = 0
        for chromosome in chromosomes:
            current += chromosome.fitness
            if current > pick:
                return chromosome

    def sus(self, players, num_players):
        sum_of_fitness = np.sum([x.fitness for x in players])
        step_size = sum_of_fitness / num_players

        # creating the ruler
        ruler = np.arange(num_players) * step_size
        random_number = np.random.uniform(0, step_size)
        ruler = ruler + random_number

        selected_players = []
        for r in ruler:
            i = 0
            f = 0
            while f < r:
                # print('{} - {}'.format(f, f + players[i]))
                f += players[i].fitness
                i += 1
            selected_players.append(players[i - 1])
        return selected_players


    def generate_new_population(self, num_players, prev_players=None):
        print('Generating New Population')
        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # # simple method
            # prob = [x.fitness for x in prev_players]
            # max_p = sum(prob)
            # prob = [p / max_p for p in prob]
            # res = np.random.choice(prev_players, num_players, p=prob, replace=False)
            # new_players = []
            # for p in res:
            #     child = copy.deepcopy(p)
            #     self.mutate(child)
            #     new_players.append(child)
            # return res.tolist()

            # TODO (additional): a selection method other than `fitness proportionate`

            # # roulette wheel method
            # new_players = []
            # count = 0
            # while count < num_players:
            #     p = self.roulette_wheel(prev_players)
            #     child = copy.deepcopy(p)
            #     self.mutate(child)
            #     new_players.append(child)
            #     count += 1
            # return new_players

            # sus method
            new_players = []
            candidates = self.sus(prev_players, num_players)
            for c in candidates:
                child = copy.deepcopy(c)
                self.mutate(child)
                new_players.append(child)
            return new_players

            # TODO (additional): implementing crossover

    def next_population_selection(self, players, num_players):
        # num_players example: 100
        # players: an array of `Player` objects
        players.sort(key=lambda x: x.fitness, reverse=True)

        # TODO (additional): a selection method other than `top-k`

        # TODO (additional): plotting
        fitnesses = np.array([x.fitness for x in players])
        max_fitness = np.max(fitnesses)
        min_fitness = np.min(fitnesses)
        avg_fitness = round(np.average(fitnesses))
        print(' - max: {}\n - min: {}\n - avg: {}'.format(max_fitness, min_fitness, avg_fitness))
        # writing statistic data on a csv file
        with open(r'fitness_data_for_plotting.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([min_fitness, avg_fitness, max_fitness])
        return players[: num_players]

