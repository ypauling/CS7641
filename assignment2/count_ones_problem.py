from opt_base_classes import OptimizationProblem
from opt_base_classes import NeighborProblem
from opt_base_classes import GAProblem
from opt_base_classes import MIMICProblem
from opt_base_classes import HillClimbingAlgorithm
from opt_base_classes import SimulatedAnnealingAlgorithm
from opt_base_classes import GeneticAlgorithm
from opt_base_classes import MIMICAlgorithm
from opt_base_classes import BayesDistTreeBinary

import numpy as np
import time
import matplotlib.pyplot as plt


class CountOnesProblem(OptimizationProblem):

    def __init__(self, length):
        super().__init__()
        self.length = length

    def fitness(self, x):
        assert type(x) == np.ndarray
        assert x.ndim == 1 or x.ndim == 2
        if x.ndim == 1:
            return np.sum(x == 1)
        else:
            return np.sum(x == 1, axis=1)


class CountOnesNeighborProblem(CountOnesProblem, NeighborProblem):

    def __init__(self, length):
        CountOnesProblem.__init__(self, length)

    def neighbor(self, x):
        i = np.random.randint(0, self.length)
        temp = x.copy()
        temp[i] = np.absolute(1 - temp[i])
        return temp

    def random_start(self):
        return np.random.randint(0, 2, size=self.length)


class CountOnesGAProblem(CountOnesProblem, GAProblem):

    def __init__(self, length, mp, cp):
        CountOnesProblem.__init__(self, length)
        self.mp = mp
        self.cp = cp

    def random_init_population(self, size):
        population = np.random.randint(0, 2, size=(size, self.length))
        return population

    def mutate(self, twins):
        # input is a size * length * 2 array
        temp = twins.copy()
        sites = np.random.binomial(1, self.mp, size=temp.shape)
        if len(np.where(sites == 1)[0]) == 0:
            return temp
        else:
            temp[np.where(sites == 1)] = 1 - temp[np.where(sites == 1)]
            return temp

    def mate(self, parents):
        twins = self.crossover(parents)
        return twins

    def crossover(self, parents):
        rec_sites = np.random.binomial(
            1, self.cp,
            size=(parents.shape[0], parents.shape[1], 1))
        rec_sites_plus = np.concatenate((rec_sites, 1-rec_sites), axis=2)
        twins = np.take_along_axis(parents, rec_sites_plus, axis=2)
        return twins


class CountOnesMIMICProblem(CountOnesProblem, MIMICProblem):

    def __init__(self, length):
        CountOnesProblem.__init__(self, length)

    def random_init_samples(self, size):
        samples = np.random.randint(0, 2, size=(size, self.length))
        return samples

    def build_distribution(self, samples):
        dist_tree = BayesDistTreeBinary(samples)
        dist_tree.generate_bayes_tree()
        return dist_tree


def fixed_iterator(n_iter, func, quiet=True):
    items = []
    values = []
    print('Start!...')
    for i in range(n_iter):
        if not quiet:
            print('move', i)
        best_value, best_item = func()
        values.append(best_value)
        items.append(best_item)
    print('Finished!...')

    return values, items


def evaluate_algorithm(algorithm, goal, max_iters, eps=1e-4):

    iter_counter = 0
    final_value = 0.
    time_spent = 0.

    time_start = time.time()
    for i in range(max_iters):
        iter_counter += 1
        final_value, _ = algorithm.move()
        if np.absolute(final_value - goal) <= eps:
            break
    time_spent = time.time() - time_start
    return final_value, iter_counter, time_spent


def print_comparisons(result, goal):
    print('goal: {}'.format(goal))
    print('{:>20s}{:>16s}{:>16s}{:>16s}'.format(
        'algorithm', 'final value', '#iterations', 'time spent')
    )
    print('{:>20s}{:>16d}{:>16d}{:>16.4f}'.format(
        'hill climbing', int(result[0, 0]), int(result[0, 1]), result[0, 2]))
    print('{:>20s}{:>16d}{:>16d}{:>16.4f}'.format(
        'simualted anealing', int(result[1, 0]),
        int(result[1, 1]), result[1, 2]))
    print('{:>20s}{:>16d}{:>16d}{:>16.4f}'.format(
        'genetic algorithm', int(result[2, 0]),
        int(result[2, 1]), result[2, 2]))
    print('{:>20s}{:>16d}{:>16d}{:>16.4f}'.format(
        'MIMIC', int(result[3, 0]), int(result[3, 1]), result[3, 2]))


def plot_result(data, filename, xarray, index, title, xlab, ylab):
    hill_data = data[0, index, :, :]
    hill_data_mean = np.mean(hill_data, axis=1)
    hill_data_std = np.std(hill_data, axis=1)

    sa_data = data[1, index, :, :]
    sa_data_mean = np.mean(sa_data, axis=1)
    sa_data_std = np.std(sa_data, axis=1)

    ga_data = data[2, index, :, :]
    ga_data_mean = np.mean(ga_data, axis=1)
    ga_data_std = np.std(ga_data, axis=1)

    mimic_data = data[3, index, :, :]
    mimic_data_mean = np.mean(mimic_data, axis=1)
    mimic_data_std = np.std(mimic_data, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.fill_between(
        xarray, hill_data_mean - hill_data_std,
        hill_data_mean + hill_data_std,
        color='red', alpha=0.5
    )
    plt.fill_between(
        xarray, sa_data_mean - sa_data_std,
        sa_data_mean + sa_data_std,
        color='green', alpha=0.5
    )
    plt.fill_between(
        xarray, ga_data_mean - ga_data_std,
        ga_data_mean + ga_data_std,
        color='blue', alpha=0.5
    )
    plt.fill_between(
        xarray, mimic_data_mean - mimic_data_std,
        mimic_data_mean + mimic_data_std,
        color='cyan', alpha=0.5
    )
    plt.plot(xarray, hill_data_mean, 'o-', color='red', label='hill')
    plt.plot(xarray, sa_data_mean, 'o-', color='green', label='annealing')
    plt.plot(xarray, ga_data_mean, 'o-', color='blue', label='GA')
    plt.plot(xarray, mimic_data_mean, 'o-', color='cyan', label='MIMIC')

    plt.legend(loc='best')
    plt.savefig(filename, format='png')
    plt.close()


n_trials = 5
problem_sizes = [20, 30, 40, 50, 60]
plot_data = np.zeros((4, 3, len(problem_sizes), n_trials), dtype=np.float)

print('Start Hill Climbing...')
for i, sz in enumerate(problem_sizes):
    problem = CountOnesNeighborProblem(sz)
    for j in range(n_trials):
        algorithm = HillClimbingAlgorithm(problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, sz, 10000
        )
        plot_data[0, 0, i, j] = final_value
        plot_data[0, 1, i, j] = iter_counter
        plot_data[0, 2, i, j] = time_spent
print('Finished...')

print('Start Simulated Annealing...')
for i, sz in enumerate(problem_sizes):
    problem = CountOnesNeighborProblem(sz)
    for j in range(n_trials):
        algorithm = SimulatedAnnealingAlgorithm(10, 0.95, problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, sz, 10000
        )
        plot_data[1, 0, i, j] = final_value
        plot_data[1, 1, i, j] = iter_counter
        plot_data[1, 2, i, j] = time_spent
print('Finished...')

print('Start Genetic Algorithm...')
for i, sz in enumerate(problem_sizes):
    problem = CountOnesGAProblem(4*sz, 0.6, 0.4)
    for j in range(n_trials):
        algorithm = GeneticAlgorithm(problem, 4*sz)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, sz, 10000
        )
        plot_data[2, 0, i, j] = final_value
        plot_data[2, 1, i, j] = iter_counter
        plot_data[2, 2, i, j] = time_spent
print('Finished...')

print('Start MIMIC...')
for i, sz in enumerate(problem_sizes):
    problem = CountOnesMIMICProblem(sz)
    for j in range(n_trials):
        algorithm = MIMICAlgorithm(4*sz, 0.75, problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, sz, 1000
        )
        plot_data[3, 0, i, j] = final_value
        plot_data[3, 1, i, j] = iter_counter
        plot_data[3, 2, i, j] = time_spent
print('Finished...')

print('Start Plotting...')
plot_result(
    plot_data, 'figures/count_one_fitness.png',
    problem_sizes, 0,
    'Count Ones Problem Fitness',
    'Sample Size', 'Average Fitness'
)
plot_result(
    plot_data, 'figures/count_one_iterations.png',
    problem_sizes, 1,
    'Count Ones Problem #Iterations',
    'Sample Size', 'Average Iterations'
)
plot_result(
    plot_data, 'figures/count_one_time.png',
    problem_sizes, 2,
    'Count Ones Problem Time Spent',
    'Sample Size', 'Average Time'
)
print('Done...')
