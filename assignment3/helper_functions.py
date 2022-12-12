import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# from sklearn import random_projection
from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

import itertools


def gaussian_confidence_2D(ax, center, covs, n_std, color):

    evals, evecs = np.linalg.eigh(covs)
    order = evals.argsort()[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    theta = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(evals)
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=theta,
        fill=False,
        edgecolor=color
    )

    ax.add_artist(ellipse)
    return ellipse


def get_covmats(covs, nf, nc, method):

    covmats = np.zeros((nc, nf, nf))
    if method == 'spherical':
        covmats = np.stack(list(np.diag(np.repeat(x, nf)) for x in covs))
    elif method == 'tied':
        covmats = np.repeat(covs[np.newaxis, :], nc, axis=0)
    elif method == 'diag':
        covmats = np.stack(list(np.diag(x) for x in covs))
    elif method == 'full':
        covmats = covs
    else:
        print('WARNING: unrecognized GMM'
              'covariance est method {}'.format(method))
    return covmats


def plot_kmeans_2D(x, y, true_labels, preds, centers, NUM_CLASS,
                   K, label_map, title, fname):
    '''
    Helper function to plot the kmeans

    Parameters:
    x: First component for plotting (after projection)
    y: Second component for plotting (after projection)
    true_labels: the correct labels for each sample
    preds: the predicted labels for each sample

    Return: None
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    markers = np.where((true_labels - preds) == 0, 0, 1)
    colors = cm.get_cmap('jet')(np.linspace(0, 1, NUM_CLASS))

    for i in range(NUM_CLASS):
        x_ma = x[(true_labels == i) & (markers == 0)]
        y_ma = y[(true_labels == i) & (markers == 0)]
        x_mi = x[(true_labels == i) & (markers == 1)]
        y_mi = y[(true_labels == i) & (markers == 1)]

        ax.plot(x_ma, y_ma, 'o', color=colors[i], alpha=0.5,
                ms=3, label='{}_matched'.format(i))
        ax.plot(x_mi, y_mi, 'x', color=colors[i], alpha=0.5,
                ms=3, label='{}_mismatched'.format(i))

    for i in range(K):
        ax.plot(centers[i, 0], centers[i, 1], 'D',
                ms=10, color=colors[label_map[i]])

    ax.legend(loc='best', fontsize='x-small')
    # plt.show()
    plt.savefig(fname, format='png')
    plt.close()


def plot_em_2D(x, y, true_labels, preds, centers, covs, NUM_CLASS,
               K, label_map, title, fname):
    '''
    Helper function to plot the EM

    Parameters:
    x: First component for plotting (after projection)
    y: Second component for plotting (after projection)
    true_labels: the correct labels for each sample
    preds: the predicted labels for each sample

    Return: None
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    markers = np.where((true_labels - preds) == 0, 0, 1)
    colors = cm.get_cmap('jet')(np.linspace(0, 1, NUM_CLASS))

    for i in range(NUM_CLASS):
        x_ma = x[(true_labels == i) & (markers == 0)]
        y_ma = y[(true_labels == i) & (markers == 0)]
        x_mi = x[(true_labels == i) & (markers == 1)]
        y_mi = y[(true_labels == i) & (markers == 1)]

        ax.plot(x_ma, y_ma, 'o', color=colors[i], alpha=0.5,
                ms=3, label='{}_matched'.format(i))
        ax.plot(x_mi, y_mi, 'x', color=colors[i], alpha=0.5,
                ms=3, label='{}_mismatched'.format(i))

    for i in range(K):
        ax.plot(centers[i, 0], centers[i, 1], 'D',
                ms=10, color=colors[label_map[i]])
        gaussian_confidence_2D(ax, centers[i, :], covs[i, :, :], 1.96,
                               colors[label_map[i]])

    ax.legend(loc='best', fontsize='x-small')
    # plt.show()
    plt.savefig(fname, format='png')
    plt.close()


def match_labels(true_labels, preds, NUM_CLASS, K):
    '''
    Helper function to match true labels and predicted labels
    The idea is to match the two groups so that the number of
    mismatches is the minimal

    Parameters:
    true_labels: an numpy array of true labels
    preds: an numpy array of predictions

    Return:
    An numpy array of relabeled predictions
    '''
    match_matrix = np.zeros((K, NUM_CLASS), dtype=np.int)
    new_labels = np.zeros(len(preds), dtype=np.int)

    for i in range(K):
        labels_i = true_labels[preds == i]
        counts = np.histogram(labels_i, bins=np.arange(-0.5, NUM_CLASS+0.5))
        counts = counts[0]
        match_matrix[i, ] = counts

    label_map = np.argmax(match_matrix, axis=1)
    error_matrix = np.sum(match_matrix, axis=1)[:, np.newaxis] - match_matrix

    for i in range(K):
        new_labels[preds == i] = label_map[i]
    min_error = np.sum(error_matrix[range(K), label_map])
    return new_labels, min_error, label_map


def calculate_distortions(K, X, seed=42):
    '''
    Function to calculate the distortions used for elbow plot
    in Kmeans analysis

    Parameter:
    K: the number of clusters to use
    X: data for clustering
    seed: the random state to use for stability

    Return:
    [float]: the distortion value
    '''
    est = KMeans(n_clusters=K, n_init=10, random_state=seed)
    est.fit(X)
    return est.inertia_


def plot_distortions(Ks, distortions, title, filename):
    '''
    Helper function to plot elbow plot
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('K')
    ax.set_ylabel('Distortions')
    ax.plot(Ks, distortions, 'o-', color='blue')
    plt.savefig(filename, format='png')
    plt.close()


def generate_and_plot_silhouette(Ks, X, title, filename, method='KMeans',
                                 ncols=5, seed=42, xleft=-0.5):
    '''
    Helper function to plot silhouette scores in K-means
    '''
    ncols = ncols
    nrows = int(np.ceil(len(Ks) / ncols))
    sil_scores = np.zeros(len(Ks), dtype=np.float)

    _, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        sharex='col', sharey='row',
        figsize=(3*ncols, 3*nrows)
    )
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]
    for i, j in itertools.product(range(nrows), range(ncols)):
        if i * ncols + j >= len(Ks):
            axes[i, j].set_axis_off()
            continue
        K = Ks[i * ncols + j]
        # print(K)
        ax = axes[i, j]

        func = KMeans
        if method == 'EM':
            func = GaussianMixture

        est = func(K, random_state=seed)
        labels = est.fit_predict(X)
        score = silhouette_score(X, labels)
        values = silhouette_samples(X, labels)
        sil_scores[i * ncols + j] = score
        colors = cm.get_cmap('jet')(np.linspace(0, 1, K))

        ax.set_xlim([xleft, 1.0])
        ax.set_ylim([0, len(X) + (K + 1) * 10])
        y_lower = 10
        for k in range(K):
            kth_values = values[labels == k]
            kth_values.sort()
            size_k = kth_values.shape[0]
            y_upper = y_lower + size_k
            color = colors[k]

            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, kth_values,
                facecolor=color, edgecolor=color, alpha=0.5
            )
            # ax.text(-0.05, y_lower + 0.5 * size_k, str(k))
            y_lower = y_upper + 10
        ax.axvline(x=score, color='red', linestyle='--')
        ax.set_title('K = {}'.format(str(k+1)))
        ax.set_xlabel('silhouette scores')
        ax.set_ylabel('Clusters')
    plt.suptitle(title)

    plt.savefig(filename, format='png')
    plt.close()
    return sil_scores


def generate_and_plot_bic_em(Ks, X, title, filename, seed=42):
    '''
    Helper function to plot BIC for Gaussian Mixture
    '''
    bic_scores = np.zeros((len(Ks), 4), dtype=np.float)
    cov_types = ['spherical', 'tied', 'diag', 'full']

    for i, K in enumerate(Ks):
        for j, cov_t in enumerate(cov_types):
            est = GaussianMixture(
                n_components=K,
                covariance_type=cov_t,
                random_state=seed
            )
            est.fit(X)
            bic_scores[i, j] = est.bic(X)

    plt.figure(figsize=(1.5*len(Ks), 1.5*len(Ks)/2))
    ax = plt.subplot(111)
    colors = cm.get_cmap('jet')(np.linspace(0, 1, 4))

    bars = []
    for i, (cov_t, color) in enumerate(zip(cov_types, colors)):
        xpos = np.array(Ks) + 0.2 * (i - 2)
        bars.append(plt.bar(xpos, bic_scores[:, i], width=0.2, color=color))
    plt.xticks(Ks)
    plt.ylim([
        np.min(bic_scores)*1.01 - 0.01*np.max(bic_scores),
        np.max(bic_scores)
    ])
    ax.set_xlabel('K')
    ax.set_ylabel('BIC scores')
    ax.legend([b[0] for b in bars], cov_types)

    ax.set_title(title)
    plt.savefig(filename, format='png')
    plt.close()
    return bic_scores
