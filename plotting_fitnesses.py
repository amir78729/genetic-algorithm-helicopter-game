import matplotlib.pyplot as plt
import csv
import numpy as np
# from game import selection_method


def plot_fitness():
    plt.style.use('dark_background')
    rows = []
    # reading from file
    with open('fitness_data_for_plotting.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        x = 1
        for row in csv_reader:
            if row:
                rows.append(row)
                plt.axvline(x, c='#232323', linestyle='dashed')
                x += 1
    try:
        # converting to numpy array
        rows = np.array(rows)
        rows = rows.astype(np.float)
        generation = np.arange(rows.shape[0]) + 1
        min_fitness = rows[:, 0]
        avg_fitness = rows[:, 1]
        max_fitness = rows[:, 2]

        # plotting
        plt.plot(generation, max_fitness, c='g', label='$fitness_{maximum}$')
        plt.fill_between(generation, max_fitness, 0, where=(max_fitness > 195), facecolor='g', alpha=0.2)

        plt.plot(generation, avg_fitness, c='y', label='$fitness_{average}$')
        plt.fill_between(generation, avg_fitness, 0, where=(avg_fitness > 195), facecolor='y', alpha=0.2)

        plt.plot(generation, min_fitness, c='r', label='$fitness_{minimum}$')
        plt.fill_between(generation, min_fitness, 0, where=(min_fitness > 195), facecolor='r', alpha=0.2)

        plt.legend()
        plt.xlabel('GENERATION')
        plt.ylabel('FITNESS')
        plt.title('Fitness of Different Generations')
        plt.show()
    except IndexError:
        print('csv file is empty!')
