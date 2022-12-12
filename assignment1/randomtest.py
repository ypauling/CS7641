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
folder = 'digits_recognition/figures/'

# load digits dataset
X, y = datasets.load_digits(return_X_y=True)
print(X.shape)
print(y.shape)


def plot_sklearn_learning_curve(solver, X, y, k, title, fname):
    '''
    A helper function to plot sklearn learning curve using CV

    Parameter:
    solver: an object implementing <fit> and <predict>
    X: training features
    y: training labels for classification
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

    # print(score_before, score_after)
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


X_tree_tr, X_tree, y_tree_tr, y_tree = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_tree_va, X_tree_tt, y_tree_va, y_tree_tt = train_test_split(
    X_tree, y_tree, test_size=0.5, random_state=42)

svm_scaler = StandardScaler()
svm_scaler.fit(X)
X_svm = svm_scaler.transform(X)

niters = [10, 20, 30, 40, 50, 100, 200, 400, 800]
solver_svm_linear = SVC(kernel='linear')
plot_score_vs_niters(solver_svm_linear, X_svm, y, niters, 3,
                     'SVM (linear kernel)',
                     'svm_linear_iter_learning_curve.png')

solver_svm_poly = SVC(kernel='poly', max_iter=-1, degree=2, gamma='scale')
plot_sklearn_learning_curve(solver_svm_poly, X_svm, y, 3,
                            'SVM (polynomial kernel)', 'svm_poly_sklearn_learning_curve.png')

niters = [10, 20, 30, 40, 50, 100, 200, 400, 800]
solver_svm_poly = SVC(kernel='poly', max_iter=-1, degree=2, gamma='scale')
plot_score_vs_niters(solver_svm_poly, X_svm, y, niters, 3,
                     'SVM (polynomial kernel)',
                     'svm_poly_iter_learning_curve.png')
