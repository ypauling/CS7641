import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time

import sklearn
import sklearn.datasets as datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Some useful constants
folder = 'breast_cancer/figures/'

# load breast cancer dataset
X, y = datasets.load_breast_cancer(return_X_y=True)
print(X.shape)
print(y.shape)


# function to plot cross-validated learning curves
def plot_sklearn_learning_curve(solver, X, y, k, title, fname):
    '''
    A helper function to plot sklearn learning curve using CV

    Parameter:
    solver: an object implementing <fit> and <predict>
    X: training features
    y: training labels for classification
    k: K fold used in cross validation
    title: titel to be used in the plot
    fname: file name to be used for the plot

    Return:
    None
    '''
    full_name = folder + fname
    train_szs, train_scores, test_scores = learning_curve(
        solver, X, y, train_sizes=np.linspace(0.2, 1.0, 5), cv=k
    )

    tr_means = np.mean(train_scores, axis=1)
    tr_stdev = np.std(train_scores, axis=1)
    tt_means = np.mean(test_scores, axis=1)
    tt_stdev = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Mean Accuracy Score')

    plt.fill_between(train_szs, tr_means-tr_stdev,
                     tr_means+tr_stdev, color='r', alpha=0.1)
    plt.fill_between(train_szs, tt_means-tt_stdev,
                     tt_means+tt_stdev, color='g', alpha=0.1)
    plt.plot(train_szs, tr_means, 'o-', color='r', label='train')
    plt.plot(train_szs, tt_means, 'o-', color='g', label='test')
    plt.legend(loc='best')

    plt.savefig(full_name, format='png')
    plt.close()


# function to calculate mean accuracy @ given iterations
def score_at_iter(solver, cv_iter, k, niter, X, y):
    '''
    A helper function to calculate cross-validated scores on
    both splited training sets and test set using a specified
    iteration number

    Parameter:
    solver: an object implementing <fit> and <score> functions
    cv_iter: cross_validation iterator used to split the dataset
    k: number of fold
    niter: number of iterations used in iterative algorithms
    X: training features
    y: training labels

    Return:
    numpy array [2, k]: a 2xk numpy array containing scores for
                        each cv training set and test set
    '''

    ret = np.zeros((2, k), dtype=np.float32)
    i = 0
    for trid, ttid in cv_iter:
        tr_dat, tt_dat = X[trid], X[ttid]
        tr_lab, tt_lab = y[trid], y[ttid]

        # fit the data
        solver.max_iter = niter
        solver.random_state = 1
        solver.fit(tr_dat, tr_lab)

        ret[:, i] = np.array(
            [solver.score(tr_dat, tr_lab), solver.score(tt_dat, tt_lab)])

        i = i + 1

    return ret


# function to plot mean accuracy v.s. iterations
def plot_score_vs_niters(solver, X, y, niters, k, title, fname):
    '''
    A function to plot estimated mean accuracy v.s. #iterations.
    However, since sklearn does not provide a way to access score
    for each iteration directly, one needs to tweak around to get
    the data.

    Parameter:
    solver: an object implementing <fit> and <predict>
    X: training features
    y: training labels
    niters: number of iterations to be used
    k: cross validation k value
    type: type of the solver, must be either 'nn' or 'svm'
    title: title to be used in the plot
    fname: file name to be used for the plot

    Return
    None
    '''

    full_name = folder + fname

    tr_scores = np.zeros((len(niters), k), dtype=np.float32)
    tt_scores = np.zeros((len(niters), k), dtype=np.float32)

    j = 0
    for i in niters:
        cv_iter = KFold(n_splits=k, random_state=42).split(X)
        result = score_at_iter(solver, cv_iter, k, i, X, y)
        tr_scores[j, :] = result[0, :]
        tt_scores[j, :] = result[1, :]
        j = j + 1

    tr_means = np.mean(tr_scores, axis=1)
    tr_stdev = np.std(tr_scores, axis=1)
    tt_means = np.mean(tt_scores, axis=1)
    tt_stdev = np.std(tt_scores, axis=1)

    plt.figure()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean Accuracy Score')
    plt.title(title)

    plt.fill_between(niters, tr_means-tr_stdev,
                     tr_means+tr_stdev, color='r', alpha=0.1)
    plt.fill_between(niters, tt_means-tt_stdev,
                     tt_means+tt_stdev, color='g', alpha=0.1)
    plt.plot(niters, tr_means, 'o-', color='r', label='train')
    plt.plot(niters, tt_means, 'o-', color='g', label='test')
    plt.legend(loc='best')

    plt.savefig(full_name, format='png')
    plt.close()


# A helper function to perform the simplest post pruning algorithm
# reduced error pruning
TREE_LEAF = -1


def prune_subtree(tree, index, X, y):
    '''
    This function uses a greedy version of reduce error pruning
    If it is better to replace any subtree with a leaf, the function
    will do so in-place and move on until all subtress are exhausted

    Parameters:
    tree: a decision tree classifier object
    index: the index of the subtree to be considred
    X: the validation dataset
    y: the validation labels

    Return:
    None
    '''
    left_child = tree.tree_.children_left[index]
    right_child = tree.tree_.children_right[index]

    if left_child != TREE_LEAF:
        prune_subtree(tree, left_child, X, y)
    if right_child != TREE_LEAF:
        prune_subtree(tree, right_child, X, y)

    score_before = tree.score(X, y)

    tree.tree_.children_left[index] = TREE_LEAF
    tree.tree_.children_right[index] = TREE_LEAF

    score_after = tree.score(X, y)

    if score_before > score_after:
        tree.tree_.children_left[index] = left_child
        tree.tree_.children_right[index] = right_child


def tree_reduced_error_pruning(tree, X, y):
    '''
    A wrapper function for the reduced error pruning algorithm
    Parameters:
    Please see in prune_subtree() function

    Return:
    None
    '''
    prune_subtree(tree, 0, X, y)


# Neural Network
# Compare different number of nodes
print('Start fitting neural network...')
nnodes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100]
result = np.zeros((len(nnodes), 3), dtype=np.float32)
for i, s in enumerate(nnodes):
    solver_nn = MLPClassifier((s, ), max_iter=2000,
                              activation='logistic', solver='lbfgs', alpha=0.1)
    result[i, :] = cross_val_score(solver_nn, X, y, cv=3)

nn_cv_mean = np.mean(result, axis=1)
nn_cv_stds = np.std(result, axis=1)

print('Start generating Figure 1...')
plt.figure()
plt.title('Neural Network v.s. #nodes')
plt.xlabel('number of nodes')
plt.ylabel('Mean Accuracy Score')
plt.fill_between(nnodes, nn_cv_mean-nn_cv_stds,
                 nn_cv_mean+nn_cv_stds, color='r', alpha=0.2)
plt.plot(nnodes, nn_cv_mean, 'o-', color='r')
plt.savefig(folder + 'nn_vs_nnodes.png', format='png')
plt.close()

print('Comparing different optimization methods...')
solver_nn = MLPClassifier((25, ), max_iter=2000,
                          activation='logistic', solver='lbfgs', alpha=0.1)
print('accuracy lbfgs:')
print(np.mean(cross_val_score(solver_nn, X, y, cv=3)))
solver_nn = MLPClassifier((25, ), max_iter=2000,
                          activation='logistic', solver='sgd', alpha=0.1)
print('accuracy sgd:')
print(np.mean(cross_val_score(solver_nn, X, y, cv=3)))
solver_nn = MLPClassifier((25, ), max_iter=2000,
                          activation='logistic', solver='adam', alpha=0.1)
print('accuracy adam:')
print(np.mean(cross_val_score(solver_nn, X, y, cv=3)))

print('Start generating Figure 2...')
solver_nn = MLPClassifier((25, ), max_iter=2000,
                          activation='logistic', solver='lbfgs', alpha=0.1)
plot_sklearn_learning_curve(solver_nn, X, y, 3,
                            'neural network', 'nn_sklearn_learning_curve.png')
print('Start generating Figure 3...')
niters = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 200, 300, 500, 1000]
solver_nn = MLPClassifier((25, ), max_iter=2000,
                          activation='logistic', solver='lbfgs', alpha=0.1)
plot_score_vs_niters(solver_nn, X, y, niters, 5,
                     'neural network', 'nn_iter_learning_curve.png')


# K nearest neighbor
# use different k values: k = [1, 3, 5]
print()
print()
print('Start fitting KNN...')

print('Start generating Figure 4...')
knn_cv_mean = np.zeros(7, dtype=np.float32)
knn_cv_stds = np.zeros(7, dtype=np.float32)
for i in [1, 2, 3, 4, 5, 6, 7]:
    solver_knn = KNeighborsClassifier(n_neighbors=i)
    values = cross_val_score(solver_knn, X, y, cv=3)
    knn_cv_mean[i-1] = np.mean(values)
    knn_cv_stds[i-1] = np.std(values)

plt.figure()
plt.title('Different k in KNN classifier')
plt.xlabel('K')
plt.ylabel('Mean Accuracy Score')
plt.fill_between([1, 2, 3, 4, 5, 6, 7],
                 knn_cv_mean - knn_cv_stds, knn_cv_mean + knn_cv_stds,
                 color='r', alpha=0.25)
plt.plot([1, 2, 3, 4, 5, 6, 7], knn_cv_mean, 'o-', color='r')
plt.savefig(folder + 'knn_diffks.png', format='png')
plt.close()

solver_knn1 = KNeighborsClassifier(n_neighbors=1)
plot_sklearn_learning_curve(solver_knn1, X, y, 3,
                            'K Nearest Neighbor (K = 1)', 'knn1_sklearn_learning_curve.png')

solver_knn3 = KNeighborsClassifier(n_neighbors=3)
plot_sklearn_learning_curve(solver_knn3, X, y, 3,
                            'K Nearest Neighbor (K = 3)', 'knn3_sklearn_learning_curve.png')

print('Start generating Figure 5...')
solver_knn6 = KNeighborsClassifier(n_neighbors=6)
plot_sklearn_learning_curve(solver_knn6, X, y, 3,
                            'K Nearest Neighbor (K = 6)', 'knn6_sklearn_learning_curve.png')


# Decision Tree
# Need some way to do the pruning
# First try the default tree, overfitting occurs
# know the boundary of the parameter search space
print()
print()
print('Start fitting Decision Tree...')

solver_tree = DecisionTreeClassifier()
solver_tree.fit(X, y)
print('Basic statistics for default tree...')
print(solver_tree.get_depth())
print(solver_tree.get_n_leaves())

# Sklearn provides ways to do some pre-pruning
# post-pruning would take some time to implement
# do a grid search over possible parameters that control the complexity of the
# tree
print('Searching for best pre-pruning condition(s)...')
tree_pre_pruning = np.zeros((7, 21, 9), dtype=np.float32)
for i in np.linspace(1, 7, 7, dtype=np.int16):
    for j in np.linspace(2, 22, 21, dtype=np.int16):
        for k in np.linspace(2, 10, 9, dtype=np.int16):
            solver_tree = DecisionTreeClassifier(
                max_depth=i, max_leaf_nodes=j, min_samples_leaf=k)
            tree_pre_pruning[i-1, j-2, k-2] = np.mean(
                cross_val_score(solver_tree, X, y, cv=4))
print('Output best combination...')
print(np.amax(tree_pre_pruning))
print(np.where(tree_pre_pruning == np.amax(tree_pre_pruning)))

# There are a few possible trees, use the simplest one
# Compare the tree with the default tree
solver_tree = DecisionTreeClassifier(random_state=42)
print('Start generating Figure 6...')
plot_sklearn_learning_curve(solver_tree, X, y, 5,
                            'Tree (before pruning)',
                            'tree_default_sklearn_learning_curve.png')
print('Accuracy from default tree...')
print(np.mean(cross_val_score(solver_tree, X, y, cv=5)))

solver_tree = DecisionTreeClassifier(
    random_state=42, max_depth=4, max_leaf_nodes=8, min_samples_leaf=8)
print('Start generating Figure 7...')
plot_sklearn_learning_curve(solver_tree, X, y, 5,
                            'Tree (after pruning)',
                            'tree_pruning_sklearn_learning_curve.png')
print('Accuracy from pre-pruning tree...')
print(np.mean(cross_val_score(solver_tree, X, y, cv=5)))

# Use a simplified post-pruning
# Split the dataset into train, validation and test
# The result is slightly better on the test dataset
print('Start post-pruning by reduced error...')
solver_tree = DecisionTreeClassifier(random_state=None)
X_tree_tr, X_tree, y_tree_tr, y_tree = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_tree_va, X_tree_tt, y_tree_va, y_tree_tt = train_test_split(
    X_tree, y_tree, test_size=0.5, random_state=42)

print('Accuracy before post-pruning...')
solver_tree.fit(X_tree_tr, y_tree_tr)
print(np.mean(solver_tree.score(X_tree_tt, y_tree_tt)))

print('Accuracy after post-pruning...')
tree_reduced_error_pruning(solver_tree, X_tree_va, y_tree_va)
print(np.mean(solver_tree.score(X_tree_tt, y_tree_tt)))


# Ada boosting ensemble method
print()
print()
print('Start fitting boosting tree...')

solver_tree = DecisionTreeClassifier(
    max_depth=4, max_leaf_nodes=8, min_samples_leaf=8)
solver_ada = AdaBoostClassifier(solver_tree)
plot_sklearn_learning_curve(solver_ada, X, y, 5,
                            'Ada Boosted Tree (before pruning)',
                            'tree_ada_before_sklearn_learning_curve.png')

# pruning
print('Searching for best pre-pruning condition(s)...')
ada_pre_pruning = np.zeros((7, 21, 9), dtype=np.float32)
for i in np.linspace(1, 7, 7, dtype=np.int16):
    for j in np.linspace(2, 22, 21, dtype=np.int16):
        for k in np.linspace(2, 10, 9, dtype=np.int16):
            solver_tree = DecisionTreeClassifier(
                max_depth=i, max_leaf_nodes=j, min_samples_leaf=k)
            solver_ada = AdaBoostClassifier(solver_tree)
            ada_pre_pruning[i-1, j-2, k-2] = np.mean(
                cross_val_score(solver_ada, X, y, cv=4))
print('Output best combination...')
print(np.amax(ada_pre_pruning))
print(np.where(ada_pre_pruning == np.amax(ada_pre_pruning)))

solver_tree = DecisionTreeClassifier(
    max_depth=6, max_leaf_nodes=10, min_samples_leaf=8)
solver_ada = AdaBoostClassifier(solver_tree)
print('Start generating Figure 8...')
plot_sklearn_learning_curve(solver_ada, X, y, 5,
                            'Ada Boosted Tree (after pruning)',
                            'tree_ada_after_sklearn_learning_curve.png')


# SVM
# use different kernels: ['linear', 'poly']
# First scale the data
print()
print()
print('Start fitting SVM...')

svm_scaler = StandardScaler()
svm_scaler.fit(X)
X_svm = svm_scaler.transform(X)

Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 200, 500, 1000]

print('Start generating Figure 9...')
result = np.zeros((len(Cs), 4), dtype=np.float32)
for i, s in enumerate(Cs):
    solver_svm = SVC(C=s, max_iter=-1, kernel='linear')
    result[i, :] = cross_val_score(solver_svm, X_svm, y, cv=4)

nn_svm_mean = np.mean(result, axis=1)
nn_svm_stds = np.std(result, axis=1)

plt.figure()
plt.title('SVM v.s. different Cs')
plt.xlabel('logC')
plt.ylabel('Mean Accuracy Score')
plt.fill_between(np.log(Cs), nn_svm_mean-nn_svm_stds,
                 nn_svm_mean+nn_svm_stds, color='r', alpha=0.2)
plt.plot(np.log(Cs), nn_svm_mean, 'o-', color='r')
plt.savefig(folder + 'svm_vs_Cs_linear.png', format='png')
plt.close()

result = np.zeros((len(Cs), 4), dtype=np.float32)
for i, s in enumerate(Cs):
    solver_svm = SVC(C=s, max_iter=-1, kernel='poly', degree=2, gamma='scale')
    result[i, :] = cross_val_score(solver_svm, X_svm, y, cv=4)

nn_svm_mean = np.mean(result, axis=1)
nn_svm_stds = np.std(result, axis=1)

plt.figure()
plt.title('SVM v.s. different Cs')
plt.xlabel('logC')
plt.ylabel('Mean Accuracy Score')
plt.fill_between(np.log(Cs), nn_svm_mean-nn_svm_stds,
                 nn_svm_mean+nn_svm_stds, color='r', alpha=0.2)
plt.plot(np.log(Cs), nn_svm_mean, 'o-', color='r')
plt.savefig(folder + 'svm_vs_Cs_poly.png', format='png')
plt.close()

solver_svm_linear = SVC(C=0.1, kernel='linear', max_iter=-1)
print('Start generating Figure 10...')
plot_sklearn_learning_curve(solver_svm_linear, X_svm, y, 3,
                            'SVM (linear kernel)', 'svm_linear_sklearn_learning_curve.png')

niters = [10, 20, 30, 40, 50, 100, 200, 400, 800]
solver_svm_linear = SVC(C=0.1, kernel='linear')
plot_score_vs_niters(solver_svm_linear, X_svm, y, niters, 3,
                     'SVM (linear kernel)',
                     'svm_linear_iter_learning_curve.png')

solver_svm_poly = SVC(C=100, kernel='poly',
                      max_iter=-1, degree=2, gamma='scale')
print('Start generating Figure 11...')
plot_sklearn_learning_curve(solver_svm_poly, X_svm, y, 3,
                            'SVM (polynomial kernel)', 'svm_poly_sklearn_learning_curve.png')

niters = [10, 20, 30, 40, 50, 100, 200, 400, 800, 1000, 2000, 5000]
solver_svm_poly = SVC(C=100, kernel='poly',
                      max_iter=-1, degree=2, gamma='scale')
plot_score_vs_niters(solver_svm_poly, X_svm, y, niters, 3,
                     'SVM (polynomial kernel)',
                     'svm_poly_iter_learning_curve.png')


# Final statistics
print()
print()
print('Final statistics...')
print('First row:     mean accuracy...')
print('Second row:    time...')
print('Neural Network')
solver_nn = MLPClassifier((25, ), max_iter=2000,
                          activation='logistic', solver='lbfgs', alpha=0.1)
t0 = time.time()
solver_nn.fit(X, y)
t1 = time.time()
print(np.mean(cross_val_score(solver_nn, X, y, cv=4)))
print(t1 - t0)

print('KNN')
solver_knn6 = KNeighborsClassifier(n_neighbors=6)
t0 = time.time()
solver_nn.fit(X, y)
t1 = time.time()
print(np.mean(cross_val_score(solver_knn6, X, y, cv=4)))
print(t1 - t0)

print('Post pruning tree')
solver_tree = DecisionTreeClassifier(random_state=None)
t0 = time.time()
solver_tree.fit(X_tree_tr, y_tree_tr)
tree_reduced_error_pruning(solver_tree, X_tree_va, y_tree_va)
t1 = time.time()
print(np.mean(solver_tree.score(X_tree_tt, y_tree_tt)))
print(t1 - t0)

print('Ada boosting tree')
solver_tree = DecisionTreeClassifier(
    max_depth=4, max_leaf_nodes=8, min_samples_leaf=8)
solver_ada = AdaBoostClassifier(solver_tree)
t0 = time.time()
solver_ada.fit(X, y)
t1 = time.time()
print(np.mean(cross_val_score(solver_ada, X, y, cv=4)))
print(t1 - t0)

print('SVM linear')
solver_svm_linear = SVC(C=0.1, kernel='linear', max_iter=-1)
t0 = time.time()
solver_svm_linear.fit(X_svm, y)
t1 = time.time()
print(np.mean(cross_val_score(solver_svm_linear, X_svm, y, cv=4)))
print(t1 - t0)

print('SVM polynomial')
solver_svm_poly = SVC(C=100, kernel='poly',
                      max_iter=-1, degree=2, gamma='scale')
t0 = time.time()
solver_svm_poly.fit(X_svm, y)
t1 = time.time()
print(np.mean(cross_val_score(solver_svm_poly, X_svm, y, cv=4)))
print(t1 - t0)
