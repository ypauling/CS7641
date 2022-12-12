from opt_base_classes import OptimizationProblem
from opt_base_classes import NeighborProblem
from opt_base_classes import GAProblem
from opt_base_classes import HillClimbingAlgorithm
from opt_base_classes import SimulatedAnnealingAlgorithm
from opt_base_classes import GeneticAlgorithm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sklearn.datasets as datasets
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class SimpleNet(nn.Module):

    def __init__(self, n_inputs, n_hiddens, n_outputs):
        super(SimpleNet, self).__init__()
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs

        self.linear1 = nn.Linear(n_inputs, n_hiddens)
        self.linear2 = nn.Linear(n_hiddens, n_outputs)

    def forward(self, X):
        X = X.view((-1, self.n_inputs))
        H = F.relu(self.linear1(X))
        return self.linear2(H)


def fitness_global(parameters, X, y, lfunc, n_hiddens):
    parameters = torch.from_numpy(parameters)
    if parameters.dim() == 1:
        parameters = parameters.reshape((1, parameters.shape[0]))
    popsize = parameters.shape[0]

    result = torch.zeros((popsize, ))
    n_inputs = 8*8
    n_outputs = 10
    W1_shape = n_inputs * n_hiddens
    W2_shape = n_hiddens * n_outputs

    for i in range(popsize):
        W1 = parameters[i, 0:W1_shape].reshape((n_inputs, n_hiddens))
        b1 = parameters[i, W1_shape:(W1_shape+n_hiddens)].reshape(
            (n_hiddens, )
        )
        W2 = parameters[
            i, (W1_shape+n_hiddens):(W1_shape+n_hiddens+W2_shape)
        ].reshape((n_hiddens, n_outputs))
        b2 = parameters[
            i, (W1_shape+n_hiddens+W2_shape):
        ].reshape((n_outputs, ))

        X = X.view((-1, n_inputs))
        H = F.relu(torch.matmul(X, W1) + b1)
        y_hat = torch.matmul(H, W2) + b2

        result[i] = lfunc(y_hat, y)

    if result.shape[0] == 1:
        return result.item()
    else:
        return result.numpy()


def fitness_global_mul(parameters, X, y, lfunc, n_hiddens):
    if parameters.ndim == 1:
        parameters = parameters.reshape((1, parameters.shape[0]))
    popsize = parameters.shape[0]

    result = np.array((popsize, ))
    n_inputs = 8*8
    n_outputs = 10
    W1_shape = n_inputs * n_hiddens
    W2_shape = n_hiddens * n_outputs

    W1 = parameters[:, 0:W1_shape].reshape(
        (popsize, n_inputs, n_hiddens)
    )
    b1 = parameters[:, W1_shape:(W1_shape+n_hiddens)].reshape(
        (popsize, n_hiddens)
    )
    W2 = parameters[
        :, (W1_shape+n_hiddens):(W1_shape+n_hiddens+W2_shape)
    ].reshape((popsize, n_hiddens, n_outputs))
    b2 = parameters[
        :, (W1_shape+n_hiddens+W2_shape):
    ].reshape((popsize, n_outputs))

    X = X.reshape((-1, n_inputs))
    H_hat = np.matmul(X, W1) + b1[:, np.newaxis, :]
    H = H_hat * (H_hat > 0)
    y_hat = np.matmul(H, W2) + b2[:, np.newaxis, :]

    result = lfunc(y_hat, y)

    if result.shape[0] == 1:
        return result.item()
    else:
        return result


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def evaluate_accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).sum().float() / y.shape[0]


def evaluate_accuracy_mul(y_hat, y):
    result = np.sum(np.argmax(y_hat, axis=2) == y, axis=1, dtype=np.float)
    return result / y_hat.shape[1]


X, y = datasets.load_digits(return_X_y=True)
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
# X = torch.from_numpy(X).float()
# y = torch.from_numpy(y)


class NeuralNetworkProblem(OptimizationProblem):

    def __init__(self, X, y, lfunc, n_hiddens, scale):
        super().__init__()
        self.imgs = X
        self.labels = y
        self.lfunc = lfunc
        self.n_hiddens = n_hiddens
        self.scale = scale

    def fitness(self, W):
        return fitness_global(
            W, self.imgs, self.labels, self.lfunc, self.n_hiddens
        )


class NeuralNetworkNeighborProblem(NeuralNetworkProblem, NeighborProblem):

    def __init__(self, X, y, lfunc, n_hiddens, scale=1):
        NeuralNetworkProblem.__init__(self, X, y, lfunc, n_hiddens, scale)

    def neighbor(self, W):
        deltas = torch.zeros_like(torch.from_numpy(W), dtype=torch.float)
        deltas.uniform_(-self.scale, self.scale)
        return W + deltas.numpy()

    def random_start(self):
        W1 = torch.zeros((64, self.n_hiddens), dtype=torch.float)
        W1 = nn.init.xavier_uniform_(W1).reshape((64*self.n_hiddens, ))
        b1 = torch.zeros((self.n_hiddens, ), dtype=torch.float)
        W2 = torch.zeros((self.n_hiddens, 10), dtype=torch.float)
        W2 = nn.init.xavier_uniform_(W2).reshape((self.n_hiddens*10, ))
        b2 = torch.zeros((10, ), dtype=torch.float)
        return torch.cat([W1, b1, W2, b2], dim=0).numpy()


class NeuralNetworkGAProblem(NeuralNetworkProblem, GAProblem):

    def __init__(self, X, y, lfunc, n_hiddens, scale=1, cp=0.5):
        NeuralNetworkProblem.__init__(self, X, y, lfunc, n_hiddens, scale)
        self.n_total = 64*n_hiddens + n_hiddens*10 + n_hiddens + 10
        self.cp = cp

    def random_init_population(self, size):
        population = torch.zeros((size, self.n_total), dtype=torch.float)
        for i in range(size):
            W1 = torch.zeros((64, self.n_hiddens), dtype=torch.float)
            W1 = nn.init.xavier_uniform_(W1).reshape((64*self.n_hiddens))
            b1 = torch.zeros((self.n_hiddens), dtype=torch.float)
            W2 = torch.zeros((self.n_hiddens, 10), dtype=torch.float)
            W2 = nn.init.xavier_uniform_(W2).reshape((self.n_hiddens*10))
            b2 = torch.zeros((10), dtype=torch.float)
            chromosome = torch.cat([W1, b1, W2, b2], dim=0)
            population[i] = chromosome
        return population.numpy()

    def mutate(self, twins):
        deltas = torch.zeros_like(torch.from_numpy(twins), dtype=torch.float)
        deltas.uniform_(-self.scale, self.scale)
        return twins + deltas.numpy()

    def mate(self, parents):
        return self.crossover(parents)

    def crossover(self, parents):
        rec_sites = np.random.binomial(
            1, self.cp,
            size=(parents.shape[0], parents.shape[1], 1)
        )
        rec_sites_plus = np.concatenate((rec_sites, 1-rec_sites), axis=2)
        twins = np.take_along_axis(parents, rec_sites_plus, axis=2)
        return twins


def train_with_BP(net, loss_func, optimizer,
                  X_tr, y_tr, X_tt, y_tt, n_epochs, scale):

    train_values = np.zeros((n_epochs//scale + 1), dtype=np.float)
    test_values = np.zeros((n_epochs//scale + 1), dtype=np.float)
    for i in range(n_epochs+1):
        with torch.enable_grad():
            optimizer.zero_grad()
            y_hat = net(X_tr)
            ltotal = loss_func(y_hat, y_tr)
            ltotal.backward()
            optimizer.step()
        with torch.no_grad():
            if i % scale == 0:
                train_values[i//scale] = evaluate_accuracy(y_hat, y_tr)
                test_values[i//scale] = evaluate_accuracy(net(X_tt), y_tt)

    return train_values, test_values


def train_with_algorithm(algorithm, X_tr, y_tr, X_tt, y_tt, n_epochs, scale):

    train_values = np.zeros((n_epochs//scale + 1), dtype=np.float)
    test_values = np.zeros((n_epochs//scale + 1), dtype=np.float)
    for i in range(n_epochs+1):
        tr_value, item = algorithm.move()
        tt_value = fitness_global(
            item, X_tt, y_tt,
            evaluate_accuracy, 15
        )
        if i % scale == 0:
            train_values[i//scale] = tr_value
            test_values[i//scale] = tt_value

    return train_values, test_values


def generate_kfold_plot_data(K, n_epochs, scale):

    kf = KFold(n_splits=K, shuffle=True)
    train_results = np.zeros((4, K, n_epochs//scale + 1), dtype=np.float)
    test_results = np.zeros((4, K, n_epochs//scale + 1), dtype=np.float)
    k = 0
    for tr_ids, tt_ids in kf.split(X):
        print('Start fold {}'.format(k+1))
        X_tr = X[tr_ids]
        X_tr_tensor = torch.from_numpy(X_tr).float()
        y_tr = y[tr_ids]
        y_tr_tensor = torch.from_numpy(y_tr)
        X_tt = X[tt_ids]
        X_tt_tensor = torch.from_numpy(X_tt).float()
        y_tt = y[tt_ids]
        y_tt_tensor = torch.from_numpy(y_tt)

        # neural network
        print('Start SGD...')
        simple_net = SimpleNet(64, 15, 10)
        simple_net.apply(init_weights)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(simple_net.parameters(), lr=0.1)
        trvs, ttvs = train_with_BP(
            simple_net, loss_function, optimizer,
            X_tr_tensor, y_tr_tensor, X_tt_tensor, y_tt_tensor,
            n_epochs, scale
        )
        train_results[0, k, :] = trvs
        test_results[0, k, :] = ttvs
        print('Finished...')

        # hill
        print('Start Hill Climbing...')
        neural_network_neighbor_problem = NeuralNetworkNeighborProblem(
            X_tr_tensor, y_tr_tensor, evaluate_accuracy, 15, scale=1
        )
        hill = HillClimbingAlgorithm(neural_network_neighbor_problem)
        hill.init_algorithm()
        trvs, ttvs = train_with_algorithm(
            hill, X_tr_tensor, y_tr_tensor,
            X_tt_tensor, y_tt_tensor, n_epochs, scale
        )
        train_results[1, k, :] = trvs
        test_results[1, k, :] = ttvs
        print('Finished...')

        # sa
        print('Start Simulated Annealing...')
        sa = SimulatedAnnealingAlgorithm(
            5, 0.95, neural_network_neighbor_problem
        )
        sa.init_algorithm()
        trvs, ttvs = train_with_algorithm(
            sa, X_tr_tensor, y_tr_tensor,
            X_tt_tensor, y_tt_tensor, n_epochs, scale
        )
        train_results[2, k, :] = trvs
        test_results[2, k, :] = ttvs
        print('Finished...')

        # ga
        print('Start Genetic Algoithm...')
        neural_network_ga_problem = NeuralNetworkGAProblem(
            X_tr_tensor, y_tr_tensor, evaluate_accuracy, 15, scale=1, cp=0.6
        )
        ga = GeneticAlgorithm(neural_network_ga_problem, 100)
        ga.init_algorithm()
        trvs, ttvs = train_with_algorithm(
            ga, X_tr_tensor, y_tr_tensor,
            X_tt_tensor, y_tt_tensor, n_epochs//10, scale//10
        )
        train_results[3, k, :] = trvs
        test_results[3, k, :] = ttvs
        print('Finished...')

        print('Fold {} done...'.format(k+1))
        print()
        k += 1

    return train_results, test_results


n_epochs = 1000000
scale = 10
trvs, ttvs = generate_kfold_plot_data(4, n_epochs, scale)

plt.figure()
plt.title('BackPropagation')
plt.xlabel('iterations')
plt.ylabel('Classification Accuracy')
xarray = np.arange(0, n_epochs//scale+1)
trvs_means = np.mean(trvs[0, :, :], axis=0)
trvs_stds = np.std(trvs[0, :, :], axis=0)
ttvs_means = np.mean(ttvs[0, :, :], axis=0)
ttvs_stds = np.std(ttvs[0, :, :], axis=0)
plt.fill_between(
    xarray, trvs_means-trvs_stds, trvs_means+trvs_stds,
    color='red', alpha=0.25
)
plt.fill_between(
    xarray, ttvs_means-ttvs_stds, ttvs_means+ttvs_stds,
    color='blue', alpha=0.25
)
plt.plot(xarray, trvs_means, 'o-', color='red', ms=0.25, label='train error')
plt.plot(xarray, ttvs_means, 'o-', color='blue',
         ms=0.25, label='validation error')
plt.legend(loc='best')
plt.savefig('figures/nn_bp_error.png', format='png')
plt.close()

plt.figure()
plt.title('Hill Climbing')
plt.xlabel('iterations')
plt.ylabel('Classification Accuracy')
xarray = np.arange(0, n_epochs//scale+1)
trvs_means = np.mean(trvs[1, :, :], axis=0)
trvs_stds = np.std(trvs[1, :, :], axis=0)
ttvs_means = np.mean(ttvs[1, :, :], axis=0)
ttvs_stds = np.std(ttvs[1, :, :], axis=0)
plt.fill_between(
    xarray, trvs_means-trvs_stds, trvs_means+trvs_stds,
    color='red', alpha=0.25
)
plt.fill_between(
    xarray, ttvs_means-ttvs_stds, ttvs_means+ttvs_stds,
    color='blue', alpha=0.25
)
plt.plot(xarray, trvs_means, 'o-', color='red', ms=0.25, label='train error')
plt.plot(xarray, ttvs_means, 'o-', color='blue',
         ms=0.25, label='validation error')
plt.legend(loc='best')
plt.savefig('figures/nn_hill_error.png', format='png')
plt.close()

plt.figure()
plt.title('Simulated Annealing')
plt.xlabel('iterations')
plt.ylabel('Classification Accuracy')
xarray = np.arange(0, n_epochs//scale+1)
trvs_means = np.mean(trvs[2, :, :], axis=0)
trvs_stds = np.std(trvs[2, :, :], axis=0)
ttvs_means = np.mean(ttvs[2, :, :], axis=0)
ttvs_stds = np.std(ttvs[2, :, :], axis=0)
plt.fill_between(
    xarray, trvs_means-trvs_stds, trvs_means+trvs_stds,
    color='red', alpha=0.25
)
plt.fill_between(
    xarray, ttvs_means-ttvs_stds, ttvs_means+ttvs_stds,
    color='blue', alpha=0.25
)
plt.plot(xarray, trvs_means, 'o-', color='red', ms=0.25, label='train error')
plt.plot(xarray, ttvs_means, 'o-', color='blue',
         ms=0.25, label='validation error')
plt.legend(loc='best')
plt.savefig('figures/nn_sa_error.png', format='png')
plt.close()

plt.figure()
plt.title('Genetic Algorithm')
plt.xlabel('iterations')
plt.ylabel('Classification Accuracy')
xarray = np.arange(0, n_epochs//scale+1)
trvs_means = np.mean(trvs[3, :, :], axis=0)
trvs_stds = np.std(trvs[3, :, :], axis=0)
ttvs_means = np.mean(ttvs[3, :, :], axis=0)
ttvs_stds = np.std(ttvs[3, :, :], axis=0)
plt.fill_between(
    xarray, trvs_means-trvs_stds, trvs_means+trvs_stds,
    color='red', alpha=0.25
)
plt.fill_between(
    xarray, ttvs_means-ttvs_stds, ttvs_means+ttvs_stds,
    color='blue', alpha=0.25
)
plt.plot(xarray, trvs_means, 'o-', color='red', ms=0.25, label='train error')
plt.plot(xarray, ttvs_means, 'o-', color='blue',
         ms=0.25, label='validation error')
plt.legend(loc='best')
plt.savefig('figures/nn_ga_error.png', format='png')
plt.close()
