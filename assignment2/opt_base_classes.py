from abc import ABC, abstractmethod
import math
import random
import numpy as np
import numpy.random
import networkx as nx
from sklearn.metrics import mutual_info_score
from scipy import stats
import time


class OptimizationProblem(ABC):

    @abstractmethod
    def fitness(self, x):
        pass


class NeighborProblem(ABC):

    @abstractmethod
    def neighbor(self, x):
        pass

    @abstractmethod
    def random_start(self):
        pass


class GAProblem(ABC):

    @abstractmethod
    def random_init_population(self, size):
        pass

    @abstractmethod
    def mutate(self, twins):
        pass

    @abstractmethod
    def mate(self, parents):
        pass

    @abstractmethod
    def crossover(self, parents):
        pass


class MIMICProblem(ABC):

    @abstractmethod
    def random_init_samples(self, size):
        pass

    @abstractmethod
    def build_distribution(self, samples):
        pass


class OptimizationAlgorithm(ABC):

    @abstractmethod
    def init_algorithm(self):
        pass

    @abstractmethod
    def move(self):
        pass


class HillClimbingAlgorithm(OptimizationAlgorithm):

    def __init__(self, neighbor_problem):
        super().__init__()
        self.problem = neighbor_problem

    def init_algorithm(self):
        self.current = self.problem.random_start()
        self.current_value = self.problem.fitness(self.current)

    def move(self):
        ngbor = self.problem.neighbor(self.current)
        ngbor_value = self.problem.fitness(ngbor)

        if ngbor_value > self.current_value:
            self.current = ngbor
            self.current_value = ngbor_value

        return self.current_value, self.current


class SimulatedAnnealingAlgorithm(OptimizationAlgorithm):

    def __init__(self, T, cooling_rate, neighbor_problem):
        super().__init__()
        self.problem = neighbor_problem
        self.T = T
        self.rate = cooling_rate

    def init_algorithm(self):
        self.current = self.problem.random_start()
        self.current_value = self.problem.fitness(self.current)

    def move(self):
        ngbor = self.problem.neighbor(self.current)
        ngbor_value = self.problem.fitness(ngbor)

        if ngbor_value > self.current_value:
            self.current = ngbor
            self.current_value = ngbor_value
        else:
            temp_value = math.exp(
                (ngbor_value - self.current_value) / (self.T+1e-6))
            if temp_value > random.random():
                self.current = ngbor
                self.current_value = ngbor_value

        self.T *= self.rate
        return self.current_value, self.current


class GeneticAlgorithm(object):

    def __init__(self, ga_problem, size):
        super().__init__()
        self.problem = ga_problem
        self.size = size

    def init_algorithm(self):
        # Assuming that the fitness function can take care of
        # a multi-dimensinal array
        self.population = self.problem.random_init_population(self.size)
        assert type(self.population) == np.ndarray
        self.values = self.problem.fitness(self.population)

    def move(self):
        weights = np.exp(self.values) / np.sum(np.exp(self.values))
        # Not exactly sure how to incorporate this part
        # sorted_chrs = np.argsort(self.values)
        # next_generation.append(self.population[sorted_chrs[-1]])
        # next_generation.append(self.population[sorted_chrs[-2]])

        # Assuming the population size is an even number
        paring = np.random.choice(self.size, size=(2, self.size//2), p=weights)
        fpool = self.population[paring[0]]
        fpool = fpool.reshape((fpool.shape[0], fpool.shape[1], 1))
        mpool = self.population[paring[1]]
        mpool = mpool.reshape((mpool.shape[0], mpool.shape[1], 1))
        parent_pool = np.concatenate((fpool, mpool), axis=2)

        twins = self.problem.mate(parent_pool)
        twins = self.problem.mutate(twins)
        self.population = self.compete(parent_pool, twins)
        self.values = self.problem.fitness(self.population)

        return np.amax(self.values), self.population[np.argmax(self.values)]

    def compete(self, ppool, cpool):
        values = np.concatenate((
            self.problem.fitness(ppool[:, :, 0])[np.newaxis, :],
            self.problem.fitness(ppool[:, :, 1])[np.newaxis, :],
            self.problem.fitness(cpool[:, :, 0])[np.newaxis, :],
            self.problem.fitness(cpool[:, :, 1])[np.newaxis, :]
        ), axis=0)
        pools = np.concatenate((ppool, cpool), axis=2)

        sorted_chrs = np.argsort(values, axis=0)
        new_population = np.concatenate((
            pools[np.arange(self.size//2), :, sorted_chrs[3]],
            pools[np.arange(self.size//2), :, sorted_chrs[2]]
        ), axis=0)
        return new_population


class MIMICAlgorithm(OptimizationAlgorithm):

    def __init__(self, size, p, mimic_problem):
        super().__init__()
        self.problem = mimic_problem
        self.size = size
        self.p = p

    def init_algorithm(self):
        self.samples = self.problem.random_init_samples(self.size)
        self.samples = np.asarray(self.samples, dtype=np.int)
        self.values = np.zeros((self.size, ), dtype=np.float)
        for i in range(self.size):
            self.values[i] = self.problem.fitness(self.samples[i])

    def move(self):
        percentile = np.percentile(self.values, self.p*100)
        cut_samples = self.samples[self.values >= percentile]
        dist = self.problem.build_distribution(cut_samples)

        self.samples = dist.sampling(self.size)
        for i in range(self.size):
            self.values[i] = self.problem.fitness(self.samples[i])

        return np.amax(self.values), self.samples[np.argmax(self.values)]


class BayesDistTree(object):

    @abstractmethod
    def generate_bayes_tree(self):
        pass

    @abstractmethod
    def sampling(self, size):
        pass


class BayesDistTreeDiscrete(BayesDistTree):

    def __init__(self, samples, N):
        super().__init__()
        self.samples = np.asarray(samples, dtype=np.int)
        self.N = N

    def compute_mutual_infos(self):
        full_graph = nx.complete_graph(self.samples.shape[1])

        for edge in full_graph.edges():
            full_graph[edge[0]][edge[1]]['weight'] = -mutual_info_score(
                self.samples[:, edge[0]], self.samples[:, edge[1]]
            )

        return full_graph

    def generate_minimum_st(self):
        full_graph = self.compute_mutual_infos()
        self.mst = nx.minimum_spanning_tree(full_graph)

    def generate_bayes_tree(self):
        self.generate_minimum_st()

        root = 0
        self.bayes_tree = nx.bfs_tree(self.mst, root)
        root_flag = 0

        for parent, child in self.bayes_tree.edges():
            pr_sample = self.samples[:, parent]
            cl_sample = self.samples[:, child]

            if root_flag == 0:
                assert parent == root
                freqs = np.histogram(pr_sample, np.arange(-0.5, self.N+0.5))[0]
                pr_density = dict(zip(range(self.N), freqs*1.0/np.sum(freqs)))
                self.bayes_tree.node[parent]['density'] = pr_density
                root_flag = 1

            for j in range(self.N):
                self.bayes_tree.node[child][j] = None
            pr_uniq = np.unique(pr_sample)
            for pr_val in pr_uniq:
                inds = np.argwhere(pr_sample == pr_val)
                cl_vals = cl_sample[inds]

                freqs = np.histogram(cl_vals, np.arange(-0.5, self.N+0.5))[0]
                cl_density = dict(zip(range(self.N), freqs*1.0/np.sum(freqs)))
                self.bayes_tree.node[child][pr_val] = cl_density

            self.bayes_tree.node[child]['density'] = \
                self.bayes_tree.node[child]

    def sampling(self, size):
        new_samples = np.zeros((size, len(self.bayes_tree.node)), np.int)

        root = self.bayes_tree.node[0]
        root_vals = list(root['density'].keys())
        root_density = list(root['density'].values())

        sample_dist = stats.rv_discrete(
            name='dist', values=(root_vals, root_density)
        )
        new_samples[:, 0] = sample_dist.rvs(size=size)

        for parent, child in nx.bfs_edges(self.bayes_tree, 0):
            cl_dict = self.bayes_tree.node[child]['density']
            for j in range(size):
                pr_val = new_samples[j, parent]
                if cl_dict[pr_val] is None:
                    new_samples[j, child] = random.randint(0, self.N-1)
                else:
                    cl_vals = list(cl_dict[int(pr_val)].keys())
                    cl_density = list(cl_dict[int(pr_val)].values())

                    sample_dist = stats.rv_discrete(
                        name='dist', values=(cl_vals, cl_density)
                    )
                    new_samples[j, child] = sample_dist.rvs(size=1)

        return new_samples


class BayesDistTreeBinary(BayesDistTree):

    def __init__(self, samples):
        super().__init__()
        self.samples = samples
        self.length = samples.shape[1]
        self.sample_size = samples.shape[0]

    def compute_mutual_infos(self):
        full_graph = nx.complete_graph(self.samples.shape[1])

        for edge in full_graph.edges():
            full_graph[edge[0]][edge[1]]['weight'] = -mutual_info_score(
                self.samples[:, edge[0]], self.samples[:, edge[1]]
            )

        return full_graph

    def generate_minimum_spanning_tree(self):
        full_graph = self.compute_mutual_infos()
        self.mst = nx.minimum_spanning_tree(full_graph)

    def generate_bayes_tree(self):
        self.generate_minimum_spanning_tree()

        root = 0
        self.bayes_tree = nx.bfs_tree(self.mst, root)

        root_sample = self.samples[:, root]
        self.root_density = np.histogram(
            root_sample, np.arange(-0.5, 2.5), density=True
        )[0]

        self.cond_densities = np.empty((self.length - 1, 2, 2))
        self.cond_densities.fill(0.5)
        all_edges = np.asarray(self.bayes_tree.edges())
        sorted_edges = all_edges[np.argsort(all_edges[:, 1])][:, 0]
        self.sorted_edges = sorted_edges

        parent_samples = self.samples[:, sorted_edges].T
        child_samples = self.samples[:, 1:].T

        child_samples_pone = child_samples * parent_samples
        n_ones = np.sum(parent_samples, axis=1)
        if np.where(n_ones != 0)[0].size != 0:
            self.cond_densities[np.where(n_ones != 0), 1, 1] = np.sum(
                child_samples_pone[np.where(n_ones != 0)], axis=1
            ) / n_ones[np.where(n_ones != 0)]
            self.cond_densities[:, 1, 0] = 1 - self.cond_densities[:, 1, 1]

        child_samples_pzero = child_samples * (1 - parent_samples)
        n_zeros = np.sum(1 - parent_samples, axis=1)
        if np.where(n_zeros != 0)[0].size != 0:
            self.cond_densities[np.where(n_zeros != 0), 0, 1] = np.sum(
                child_samples_pzero[np.where(n_zeros != 0)], axis=1
            ) / n_zeros[np.where(n_zeros != 0)]
            self.cond_densities[:, 0, 0] = 1 - self.cond_densities[:, 0, 1]

    def sampling(self, size):

        new_samples = np.zeros((size, self.length), dtype=np.int)

        root_sampler = np.random.uniform(0, 1, size=(size, ))
        new_samples[:, 0] = np.where(root_sampler > self.root_density[0], 1, 0)

        for i in np.arange(1, self.length):
            pr_sample = new_samples[:, self.sorted_edges[i-1]]
            current_density = self.cond_densities[i-1, pr_sample, 0]
            sampler = np.random.uniform(0, 1, size=(size, ))
            new_samples[:, i] = np.where(sampler > current_density, 1, 0)

        return new_samples


if __name__ == '__main__':
    samples = np.random.randint(0, 2, size=(100, 10))
    samples = np.ones((100, 10))
    dist_discrete = BayesDistTreeDiscrete(samples, 2)
    dist_binary = BayesDistTreeBinary(samples)

    time_start = time.time()
    dist_discrete.generate_bayes_tree()
    print(nx.get_node_attributes(dist_discrete.bayes_tree, 'density'))
    print('old implmentation: {:>5.2f}'.format(time.time() - time_start))

    time_start = time.time()
    dist_binary.generate_bayes_tree()
    print(dist_binary.root_density)
    print(dist_binary.cond_densities)
    print('new implmentation: {:>5.2f}'.format(time.time() - time_start))
