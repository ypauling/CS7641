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
import contextlib


def knapsack_answer(wt, val, W, n):
    Mvals = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(W + 1):
            if i == 0 or j == 0:
                Mvals[i][j] = 0
            elif wt[i-1] <= j:
                Mvals[i][j] = max(
                    val[i-1] + Mvals[i-1][j-wt[i-1]], Mvals[i-1][j]
                )
            else:
                Mvals[i][j] = Mvals[i-1][j]
    return Mvals[n][W]


class KnapsackProblem(OptimizationProblem):

    def __init__(self, object_weights, object_values, weight_constraint):
        super().__init__()
        assert type(object_weights) == np.ndarray
        self.object_weights = object_weights
        assert type(object_values) == np.ndarray
        self.object_values = object_values
        self.weight_constraint = weight_constraint

    def fitness(self, x):
        assert type(x) == np.ndarray

        x_copy = x.copy()
        if x_copy.ndim == 1:
            x_copy = x_copy[np.newaxis, :]

        total_values = x_copy.dot(self.object_values)
        total_weights = x_copy.dot(self.object_weights)

        values = np.where(
            total_weights < self.weight_constraint, total_values, 0
        )
        if values.shape[0] == 1:
            return values[0]
        else:
            return values


class KnapsackNeighborProblem(KnapsackProblem, NeighborProblem):

    def __init__(self, weights, values, constraint, length):
        KnapsackProblem.__init__(self, weights, values, constraint)
        self.length = length

    def neighbor(self, x):
        ngbor = x.copy()
        i = np.random.randint(0, self.length)
        ngbor[i] = np.absolute(1 - ngbor[i])
        return ngbor

    def random_start(self):
        return np.random.randint(0, 2, size=self.length)


class KnapsackGAProblem(KnapsackProblem, GAProblem):

    def __init__(self, weights, values, constraint, length, mp=0.5, cp=0.5):
        KnapsackProblem.__init__(self, weights, values, constraint)
        self.length = length
        self.mp = mp
        self.cp = cp

    def random_init_population(self, size):
        population = np.random.randint(0, 2, size=(size, self.length))
        return population

    def mutate(self, twins):
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


class KnapsackMIMICProblem(KnapsackProblem, MIMICProblem):

    def __init__(self, weights, values, constraint, length):
        KnapsackProblem.__init__(self, weights, values, constraint)
        self.length = length

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


def evaluate_algorithm(algorithm, goal, max_iters, eps=0.01):

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
        color='red', alpha=0.1
    )
    plt.fill_between(
        xarray, sa_data_mean - sa_data_std,
        sa_data_mean + sa_data_std,
        color='green', alpha=0.1
    )
    plt.fill_between(
        xarray, ga_data_mean - ga_data_std,
        ga_data_mean + ga_data_std,
        color='blue', alpha=0.1
    )
    plt.fill_between(
        xarray, mimic_data_mean - mimic_data_std,
        mimic_data_mean + mimic_data_std,
        color='cyan', alpha=0.1
    )
    plt.plot(xarray, hill_data_mean, 'o-', color='red', label='hill')
    plt.plot(xarray, sa_data_mean, 'o-', color='green', label='annealing')
    plt.plot(xarray, ga_data_mean, 'o-', color='blue', label='GA')
    plt.plot(xarray, mimic_data_mean, 'o-', color='cyan', label='MIMIC')

    plt.legend(loc='best')
    plt.savefig(filename, format='png')
    plt.close()


@contextlib.contextmanager
def local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


object_weight_upper = 100
object_value_upper = 20


def generate_dataset(size, scale=3):
    dataset = {}
    with local_seed(42):
        object_weights = np.random.randint(1, object_weight_upper, size=size)
        object_values = np.random.randint(1, object_value_upper, size=size)
        object_constraint = object_weight_upper * size // scale

        dataset['weights'] = object_weights
        dataset['values'] = object_values
        dataset['constraint'] = object_constraint

    return dataset


n_trials = 5
problem_sizes = [20, 30, 40, 50, 60]
plot_data = np.zeros((4, 3, len(problem_sizes), n_trials), dtype=np.float)
datasets = {}
for sz in problem_sizes:
    datasets[sz] = generate_dataset(sz)

print('Start Hill Climbing...')
for i, sz in enumerate(problem_sizes):
    problem = KnapsackNeighborProblem(
        datasets[sz]['weights'],
        datasets[sz]['values'],
        datasets[sz]['constraint'],
        sz
    )
    goal = knapsack_answer(
        datasets[sz]['weights'],
        datasets[sz]['values'],
        int(datasets[sz]['constraint']),
        int(sz)
    )
    for j in range(n_trials):
        algorithm = HillClimbingAlgorithm(problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, goal, 10000
        )
        plot_data[0, 0, i, j] = final_value
        plot_data[0, 1, i, j] = iter_counter
        plot_data[0, 2, i, j] = time_spent
print('Finished...')

print('Start Simulated Annealing...')
for i, sz in enumerate(problem_sizes):
    problem = KnapsackNeighborProblem(
        datasets[sz]['weights'],
        datasets[sz]['values'],
        datasets[sz]['constraint'],
        sz
    )
    goal = knapsack_answer(
        datasets[sz]['weights'],
        datasets[sz]['values'],
        int(datasets[sz]['constraint']),
        int(sz)
    )
    for j in range(n_trials):
        algorithm = SimulatedAnnealingAlgorithm(10, 0.95, problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, goal, 10000
        )
        plot_data[1, 0, i, j] = final_value
        plot_data[1, 1, i, j] = iter_counter
        plot_data[1, 2, i, j] = time_spent
print('Finished...')

print('Start Genetic Algorithm...')
for i, sz in enumerate(problem_sizes):
    problem = KnapsackGAProblem(
        datasets[sz]['weights'],
        datasets[sz]['values'],
        datasets[sz]['constraint'],
        sz,
        0.5,
        0.5
    )
    goal = knapsack_answer(
        datasets[sz]['weights'],
        datasets[sz]['values'],
        int(datasets[sz]['constraint']),
        int(sz)
    )
    for j in range(n_trials):
        algorithm = GeneticAlgorithm(problem, 4*sz)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, goal, 10000
        )
        plot_data[2, 0, i, j] = final_value
        plot_data[2, 1, i, j] = iter_counter
        plot_data[2, 2, i, j] = time_spent
print('Finished...')

print('Start MIMIC...')
for i, sz in enumerate(problem_sizes):
    problem = KnapsackMIMICProblem(
        datasets[sz]['weights'],
        datasets[sz]['values'],
        datasets[sz]['constraint'],
        sz
    )
    goal = knapsack_answer(
        datasets[sz]['weights'],
        datasets[sz]['values'],
        int(datasets[sz]['constraint']),
        int(sz)
    )
    for j in range(n_trials):
        algorithm = MIMICAlgorithm(4*sz, 0.75, problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, goal, 1000
        )
        plot_data[3, 0, i, j] = final_value
        plot_data[3, 1, i, j] = iter_counter
        plot_data[3, 2, i, j] = time_spent
print('Finished...')

print('Start Plotting...')
plot_result(
    plot_data, 'figures/knapsack_fitness.png',
    problem_sizes, 0,
    'Knapsack Problem Fitness',
    'Sample Size', 'Average Fitness'
)
plot_result(
    plot_data, 'figures/knapsack_iterations.png',
    problem_sizes, 1,
    'Knapsack Problem #Iterations',
    'Sample Size', 'Average Iterations'
)
plot_result(
    plot_data, 'figures/knapsack_time.png',
    problem_sizes, 2,
    'Knapsack Problem Time Spent',
    'Sample Size', 'Average Time'
)
print('Done...')
