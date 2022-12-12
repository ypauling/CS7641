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
from mpl_toolkits.mplot3d import Axes3D


def fitness_global(x, T):
    # assuming x is a numpy array
    if x.ndim == 1:
        x = x[np.newaxis, :]
    n_dims = x.shape[1]
    x_reverse = x[:, ::-1]
    x_complement = 1 - x

    tail = np.sum(
        np.where(np.cumsum(x_reverse, axis=1) == 0, 1, 0), axis=1
    )
    head = np.sum(
        np.where(np.cumsum(x_complement, axis=1) == 0, 1, 0), axis=1
    )

    tail_reward = np.where(tail > T, 1, 0)
    head_reward = np.where(head > T, 1, 0)
    reward = tail_reward * head_reward * n_dims

    values = np.amax(np.vstack((tail, head)), axis=0) + reward
    if values.size == 1:
        return values.item()
    else:
        return values


class FourPeaksProblem(OptimizationProblem):

    def __init__(self, T, n_dims):
        super().__init__()
        self.T = T
        self.n_dims = n_dims

    def fitness(self, x):
        return fitness_global(x, self.T)


class FourPeaksNeighborProblem(FourPeaksProblem, NeighborProblem):

    def __init__(self, T, n_dims):
        FourPeaksProblem.__init__(self, T, n_dims)

    def neighbor(self, x):
        ngbor = x.copy()
        i = np.random.randint(0, self.n_dims)
        ngbor[i] = 1 - ngbor[i]
        return ngbor

    def random_start(self):
        return np.random.randint(0, 2, size=self.n_dims)


class FourPeaksGAProblem(FourPeaksProblem, GAProblem):

    def __init__(self, T, n_dims, cp=0.5, mp=0.5):
        FourPeaksProblem.__init__(self, T, n_dims)
        self.mp = mp
        self.cp = cp

    def random_init_population(self, size):
        population = np.random.randint(0, 2, size=(size, self.n_dims))
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


class FourPeaksMIMICProblem(FourPeaksProblem, MIMICProblem):

    def __init__(self, T, n_dims):
        FourPeaksProblem.__init__(self, T, n_dims)

    def random_init_samples(self, size):
        samples = np.random.randint(0, 2, size=(size, self.n_dims))
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


fig = plt.figure()
_ = Axes3D(fig)
plt.close()

print('Generating Parameter Tuning Plot...')
popszs = [20, 40, 160, 500, 1000]
pts = [0.2, 0.5, 0.8, 0.9]
tuning_params = np.zeros((5, 4), dtype=np.int)
for i, popsz in enumerate(popszs):
    for j, pt in enumerate(pts):
        problem = FourPeaksMIMICProblem(20//5, 20)
        algorithm = MIMICAlgorithm(popsz, pt, problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, 35, 200
        )
        tuning_params[i, j] = final_value
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Effects of parameters')
ax.scatter(
    *np.meshgrid(pts, popszs),
    tuning_params, c='red'
)
for i, popsz in enumerate(popszs):
    for j, pt in enumerate(pts):
        ax.plot(
            [pt, pt], [popsz, popsz],
            [0, tuning_params[i, j]],
            'k--', c='red'
        )
ax.set_xlabel('percentile')
ax.set_ylabel('pop size')
ax.set_zlabel('fitness')
plt.savefig('figures/four_peaks_parameter_tuning.png', format='png')
plt.close()
print('Finished...')

n_trials = 4
n_dims = [5, 10, 15, 20, 25]
plot_data = np.zeros((4, 3, len(n_dims), n_trials), dtype=np.float)

print('Start Hill Climbing...')
for i, sz in enumerate(n_dims):
    problem = FourPeaksNeighborProblem(sz // 5, sz)
    for j in range(n_trials):
        algorithm = HillClimbingAlgorithm(problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, 2*sz - (sz//5+1), 10000
        )
        plot_data[0, 0, i, j] = final_value
        plot_data[0, 1, i, j] = iter_counter
        plot_data[0, 2, i, j] = time_spent
print('Finished...')

print('Start Simulated Annealing...')
for i, sz in enumerate(n_dims):
    problem = FourPeaksNeighborProblem(sz // 5, sz)
    for j in range(n_trials):
        algorithm = SimulatedAnnealingAlgorithm(10, 0.95, problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, 2*sz - (sz//5+1), 10000
        )
        plot_data[1, 0, i, j] = final_value
        plot_data[1, 1, i, j] = iter_counter
        plot_data[1, 2, i, j] = time_spent
print('Finished...')

print('Start Genetic Algorithm...')
for i, sz in enumerate(n_dims):
    problem = FourPeaksGAProblem(sz // 5, sz, 0.6, 0.4)
    for j in range(n_trials):
        algorithm = GeneticAlgorithm(problem, 500)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, 2*sz - (sz//5+1), 10000
        )
        plot_data[2, 0, i, j] = final_value
        plot_data[2, 1, i, j] = iter_counter
        plot_data[2, 2, i, j] = time_spent
print('Finished...')

print('Start MIMIC...')
for i, sz in enumerate(n_dims):
    problem = FourPeaksMIMICProblem(sz // 5, sz)
    for j in range(n_trials):
        algorithm = MIMICAlgorithm(1000, 0.8, problem)
        algorithm.init_algorithm()
        final_value, iter_counter, time_spent = evaluate_algorithm(
            algorithm, 2*sz - (sz//5+1), 1000
        )
        plot_data[3, 0, i, j] = final_value
        plot_data[3, 1, i, j] = iter_counter
        plot_data[3, 2, i, j] = time_spent
print('Finished...')

print('Start Plotting...')
plot_result(
    plot_data, 'figures/four_peaks_fitness.png',
    n_dims, 0,
    'Four Peaks Problem Fitness',
    'Sample Size', 'Average Fitness'
)
plot_result(
    plot_data, 'figures/four_peaks_iterations.png',
    n_dims, 1,
    'Four Peaks Problem #Iterations',
    'Sample Size', 'Average Iterations'
)
plot_result(
    plot_data, 'figures/four_peaks_time.png',
    n_dims, 2,
    'Four Peaks Problem Time Spent',
    'Sample Size', 'Average Time'
)
print('Done...')
