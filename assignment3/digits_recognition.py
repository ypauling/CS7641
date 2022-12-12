import numpy as np

from helper_functions import get_covmats
from helper_functions import plot_kmeans_2D
from helper_functions import plot_em_2D
from helper_functions import match_labels
from helper_functions import calculate_distortions
from helper_functions import plot_distortions
from helper_functions import generate_and_plot_silhouette
from helper_functions import generate_and_plot_bic_em

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import random_projection
# from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
# import matplotlib.cm as cm
from scipy.stats import kurtosis
import time

NUM_CLASS = 10
folder = 'figures/digits/'

#################
# Load the data #
#################
X_raw, y = load_digits(return_X_y=True)
X = scale(X_raw)

################
# Use raw data #
################
# KMeans on original data
# Select the best K to use for Kmeans
Ks = range(2, 30)

print('Start using raw features for K-means...')

print('Generate elbow plots...')
distortions = np.zeros(len(Ks), dtype=np.float)
for i, K in enumerate(Ks):
    distortions[i] = calculate_distortions(K, X)
plot_distortions(Ks, distortions, 'Raw Data', folder+'kmeans_raw_elbow.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X, 'Raw Data', folder+'kmeans_raw_sil.png',
    xleft=-0.2, ncols=10
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot k-means result...')
K = 21
km_est = KMeans(K, random_state=42)
y_hat = km_est.fit_predict(X)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total errors for Kmeans on raw: {}'.format(terrors))
plot_pca = PCA(2, random_state=42)
projs = plot_pca.fit_transform(X)
plot_kmeans_2D(projs[:, 0], projs[:, 1], y, y_hat,
               plot_pca.transform(km_est.cluster_centers_), NUM_CLASS, K,
               label_map, 'Raw data', folder+'kmeans_raw_result.png')
print('Done...')

print('Kmeans on raw data finished...')

# EM on original data
# Select the number of components
Ks = range(2, 30)

print()
print('Start using raw features for EM...')

print('Generate BIC bar plots...')
generate_and_plot_bic_em(Ks, X, 'Raw Data', folder+'em_raw_bic.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X, 'Raw Data', folder+'em_raw_sil.png',
    xleft=-0.2, ncols=10, method='EM'
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot EM result...')
K = 28
em_est = GaussianMixture(K, 'diag', random_state=42)
y_hat = em_est.fit_predict(X)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total errors for EM on raw: {}'.format(terrors))
plot_pca = PCA(2, random_state=42)
projs = plot_pca.fit_transform(X)
covs = get_covmats(em_est.covariances_, 64, K, 'diag')
covmats = np.matmul(
    np.matmul(plot_pca.components_, covs),
    plot_pca.components_.T
)
plot_em_2D(projs[:, 0], projs[:, 1], y, y_hat,
           plot_pca.transform(em_est.means_), covmats, NUM_CLASS, K,
           label_map, 'Raw Data', folder+'em_raw_result.png')
print('Done...')

print('EM on raw data finished...')

###############################
# Use PCA to pre-process data #
###############################
print()
print('Start using PCA for pre-processing...')

print('Plotting explained variance against K...')
K = 64
pca = PCA(K, random_state=42)
X_pca = pca.fit_transform(X)

fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('PCA variance explained')
ax1.set_xlabel('K')
ax1.set_ylabel('Percentage of variance')
ax1.plot(
    range(1, K+1), np.cumsum(pca.explained_variance_ratio_),
    'o-', color='blue'
)

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('PCA eigenvalues')
ax2.set_xlabel('K')
ax2.set_ylabel('Eigenvalues')
ax2.plot(
    range(1, K+1), pca.explained_variance_,
    'o-', color='red'
)

plt.savefig(folder+'pca_variance_explained.png', format='png')
plt.close()
print('Done')

K = np.sum(np.cumsum(pca.explained_variance_ratio_) <= 0.9)
print('Use {} components (explained var > 0.9)...'.format(K))

pca = PCA(K, random_state=42)
X_pca = pca.fit_transform(X)

# KMeans on PCA data
# Select the best K to use for Kmeans
Ks = range(2, 30)

print('Start using PCA features for K-means...')

print('Generate elbow plots...')
distortions = np.zeros(len(Ks), dtype=np.float)
for i, K in enumerate(Ks):
    distortions[i] = calculate_distortions(K, X_pca)
plot_distortions(Ks, distortions, 'PCA Data', folder+'kmeans_pca_elbow.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X_pca, 'PCA Data', folder+'kmeans_pca_sil.png',
    xleft=-0.2, ncols=10
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot k-means result...')
K = 24
km_est = KMeans(K, random_state=42)
y_hat = km_est.fit_predict(X_pca)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total error for KMeans on PCA: {}'.format(terrors))
plot_kmeans_2D(X_pca[:, 0], X_pca[:, 1], y, y_hat,
               km_est.cluster_centers_[:, 0:2], NUM_CLASS, K,
               label_map, 'PCA data', folder+'kmeans_pca_result.png')
print('Done...')

print('Kmeans on PCA data finished...')

# EM on PCA data
# Select the number of components
Ks = range(2, 30)

print()
print('Start using PCA features for EM...')

print('Generate BIC bar plots...')
generate_and_plot_bic_em(Ks, X_pca, 'PCA Data', folder+'em_pca_bic.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X_pca, 'PCA Data', folder+'em_pca_sil.png',
    xleft=-0.2, ncols=10, method='EM'
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot EM result...')
K = 9
em_est = GaussianMixture(K, 'full', random_state=42)
y_hat = em_est.fit_predict(X_pca)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total error for EM on PCA: {}'.format(terrors))
covs = get_covmats(em_est.covariances_, X_pca.shape[1], K, 'full')
covmats = covs[:, 0:2, 0:2]
plot_em_2D(X_pca[:, 0], X_pca[:, 1], y, y_hat,
           em_est.means_[:, 0:2], covmats, NUM_CLASS, K,
           label_map, 'PCA Data', folder+'em_pca_result.png')
print('Done...')

print('EM on PCA data finished...')


###############################
# Use ICA to pre-process data #
###############################
print()
print('Start using ICA for preprocessing...')

K = 60
ica = FastICA(K, random_state=42, max_iter=1000000)
X_centered = X_raw - np.mean(X_raw, axis=0)
X_centered -= np.mean(X_centered, axis=1).reshape(X.shape[0], -1)
X_ica = ica.fit_transform(X_centered)

# plot kurtosis of different components
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Kurtosis of different components')
ax1.set_xlabel('K')
ax1.set_ylabel('Kurtosis')
kur_vals = np.zeros(K, dtype=np.float)
for i in range(K):
    kur_vals[i] = kurtosis(X_ica[:, i])
ax1.plot(range(1, K+1), kur_vals, 'o-', color='blue')
plt.savefig(folder+'ica_kurtosis.png', format='png')
plt.close()

# Get the first few components that shows large kurtosis
# Show that this is too sensitive to outliers
sel_cols = np.argsort(kur_vals)[::-1][0:2]
projs = X_ica[:, sel_cols]
plt.figure()
x_control = projs[:, 0][np.where(y == 0)]
y_control = projs[:, 1][np.where(y == 0)]
x_cancer = projs[:, 0][np.where(y == 1)]
y_cancer = projs[:, 1][np.where(y == 1)]
plt.plot(
    x_control, y_control,
    'o', ms=1, color='red', label='Control'
)
plt.plot(
    x_cancer, y_cancer,
    'o', ms=1, color='blue', label='Cancer'
)
plt.legend(loc='best')
plt.savefig(folder + 'ica_kurtosis_outlier.png', format='png')
plt.close()

# Or use RFECV to select the best features
rfc = RandomForestClassifier(100, random_state=42)
selector = RFECV(rfc, cv=5)
selector = selector.fit(X_ica, y)

X_ica = X_ica[:, selector.support_]

# KMeans on ICA data
# Select the best K
Ks = range(2, 30)

print('Start using ICA features for K-means...')

print('Generate elbow plots...')
distortions = np.zeros(len(Ks), dtype=np.float)
for i, K in enumerate(Ks):
    distortions[i] = calculate_distortions(K, X_ica)
plot_distortions(Ks, distortions, 'ICA Data', folder+'kmeans_ica_elbow.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X_ica, 'ICA Data', folder+'kmeans_ica_sil.png',
    xleft=-0.2, ncols=10
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot k-means result...')
K = 10
km_est = KMeans(K, random_state=42)
y_hat = km_est.fit_predict(X_ica)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total error for KMeans on ICA: {}'.format(terrors))
plot_kmeans_2D(X_ica[:, 0], X_ica[:, 1], y, y_hat,
               km_est.cluster_centers_[:, 0:2], NUM_CLASS, K,
               label_map, 'ICA data', folder+'kmeans_ica_result.png')
print('Done...')

print('Kmeans on ICA data finished...')

# EM on ICA data
# Select the best K
Ks = range(2, 30)

print()
print('Start using ICA features for EM...')

print('Generate BIC bar plots...')
generate_and_plot_bic_em(Ks, X_ica, 'ICA Data', folder+'em_ica_bic.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X_ica, 'ICA Data', folder+'em_ica_sil.png',
    xleft=-0.2, ncols=10, method='EM'
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot EM result...')
K = 20
em_est = GaussianMixture(K, 'diag', random_state=42)
y_hat = em_est.fit_predict(X_ica)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total error for EM on ICA: {}'.format(terrors))
covs = get_covmats(em_est.covariances_, X_ica.shape[1], K, 'diag')
covmats = covs[:, 0:2, 0:2]
plot_em_2D(X_ica[:, 0], X_ica[:, 1], y, y_hat,
           em_est.means_[:, 0:2], covmats, NUM_CLASS, K,
           label_map, 'ICA Data', folder+'em_ica_result.png')
print('Done...')

print('EM on ICA data finished...')

#####################
# Random Projection #
#####################
print()
print('Start using RP for preprocessing...')

# Estimate the number of dimensions by JL Lemma
print(random_projection.johnson_lindenstrauss_min_dim(X.shape[0], 0.1))

K = 30
Niter = 100
terrors_rp = np.zeros(Niter, dtype=np.float)
for i, s in enumerate(range(100)):
    rp = random_projection.GaussianRandomProjection(
        K, random_state=s
    )
    X_rp = rp.fit_transform(X)
    rfc = RandomForestClassifier(
        n_estimators=100, max_depth=2,
        random_state=42
    )
    rfc.fit(X_rp, y)
    terrors_rp[i] = rfc.score(X_rp, y)

plt.figure()
plt.hist(terrors_rp, bins=30)
plt.savefig(folder+'rp_matcherrors_hist.png', format='png')
plt.close()

seed_used = np.argsort(terrors_rp)[::-1][0]
rp = random_projection.GaussianRandomProjection(
    K, random_state=seed_used
)
X_rp = rp.fit_transform(X)

# KMeans on RP data
# Select the best K
Ks = range(2, 30)

print('Start using RP features for K-means...')

print('Generate elbow plots...')
distortions = np.zeros(len(Ks), dtype=np.float)
for i, K in enumerate(Ks):
    distortions[i] = calculate_distortions(K, X_rp)
plot_distortions(Ks, distortions, 'RP Data', folder+'kmeans_rp_elbow.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X_rp, 'RP Data', folder+'kmeans_rp_sil.png',
    xleft=-0.2, ncols=10
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot k-means result...')
K = 13
km_est = KMeans(K, random_state=42)
y_hat = km_est.fit_predict(X_rp)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total error for KMeans on RP: {}'.format(terrors))
plot_kmeans_2D(X_rp[:, 0], X_rp[:, 1], y, y_hat,
               km_est.cluster_centers_[:, 0:2], NUM_CLASS, K,
               label_map, 'RP data', folder+'kmeans_rp_result.png')
print('Done...')

print('Kmeans on RP data finished...')

# EM on RP data
# Select the best K
Ks = range(2, 30)

print()
print('Start using RP features for EM...')

print('Generate BIC bar plots...')
generate_and_plot_bic_em(Ks, X_rp, 'RP Data', folder+'em_rp_bic.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X_rp, 'RP Data', folder+'em_rp_sil.png',
    xleft=-0.2, ncols=10, method='EM'
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot EM result...')
K = 7
em_est = GaussianMixture(K, 'full', random_state=42)
y_hat = em_est.fit_predict(X_rp)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total error for EM on RP: {}'.format(terrors))
covs = get_covmats(em_est.covariances_, X_rp.shape[1], K, 'full')
covmats = covs[:, 0:2, 0:2]
plot_em_2D(X_rp[:, 0], X_rp[:, 1], y, y_hat,
           em_est.means_[:, 0:2], covmats, NUM_CLASS, K,
           label_map, 'RP Data', folder+'em_rp_result.png')
print('Done...')

print('EM on RP data finished...')


#############################################
# Use mutual information to select features #
#############################################
# Select the best K features to use for downstream analysis
print()

print('Start selecting number of features using RFECV...')
rfc = RandomForestClassifier(100, random_state=42)
selector = RFECV(rfc, cv=5)
selector = selector.fit(X, y)

X_fs = X[:, selector.support_]
print('Done...')

# KMeans on feature selected data
# Select the best K
Ks = range(2, 30)

print('Start using RFECV features for K-means...')

print('Generate elbow plots...')
distortions = np.zeros(len(Ks), dtype=np.float)
for i, K in enumerate(Ks):
    distortions[i] = calculate_distortions(K, X_fs)
plot_distortions(Ks, distortions, 'RFECV Data', folder+'kmeans_fs_elbow.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X_fs, 'RFECV Data', folder+'kmeans_fs_sil.png',
    xleft=-0.2, ncols=10
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot k-means result...')
K = 12
km_est = KMeans(K, random_state=42)
y_hat = km_est.fit_predict(X_fs)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total error for KMeans on RFECV: {}'.format(terrors))
plot_pca = PCA(2, random_state=42)
projs = plot_pca.fit_transform(X_fs)
plot_kmeans_2D(projs[:, 0], projs[:, 1], y, y_hat,
               plot_pca.transform(km_est.cluster_centers_), NUM_CLASS, K,
               label_map, 'RFECV data', folder+'kmeans_fs_result.png')
print('Done...')

print('Kmeans on RFECV data finished...')

# EM on feature selected data
# Select the number of components
Ks = range(2, 30)

print()
print('Start using RFECV for EM...')

print('Generate BIC bar plots...')
generate_and_plot_bic_em(Ks, X_fs, 'RFECV Data', folder+'em_fs_bic.png')
print('Done...')

print('Generate Silhouette plots...')
vals = generate_and_plot_silhouette(
    Ks, X_fs, 'RFECV Data', folder+'em_fs_sil.png',
    xleft=-0.2, ncols=10, method='EM'
)
print(np.argsort(vals)[::-1][0] + 2)
print('Done...')

print('Plot EM result...')
K = 12
em_est = GaussianMixture(K, 'full', random_state=42)
y_hat = em_est.fit_predict(X_fs)
y_hat, terrors, label_map = match_labels(y, y_hat, NUM_CLASS, K)
print('Total errors for EM on RFECV: {}'.format(terrors))
plot_pca = PCA(2, random_state=42)
projs = plot_pca.fit_transform(X_fs)
covs = get_covmats(em_est.covariances_, X_fs.shape[1], K, 'full')
covmats = np.matmul(
    np.matmul(plot_pca.components_, covs),
    plot_pca.components_.T
)
plot_em_2D(projs[:, 0], projs[:, 1], y, y_hat,
           plot_pca.transform(em_est.means_), covmats, NUM_CLASS, K,
           label_map, 'RFECV Data', folder+'em_fs_result.png')
print('Done...')

print('EM on RFECV data finished...')


################################################
# Run neural network with different algorithms #
################################################
print()
print('Test neural network performances...')
nnet = MLPClassifier((15,), random_state=42, max_iter=200000)

cv_scores = np.zeros(7, dtype=np.float)
cv_stds = np.zeros(7, dtype=np.float)
run_times = np.zeros(7, dtype=np.float)

kmest = KMeans(21, random_state=42)
emest = GaussianMixture(28, 'diag', random_state=42)

X_km = kmest.fit_predict(X).reshape(-1, 1)
X_em = emest.fit_predict(X).reshape(-1, 1)

features = [X, X_pca, X_ica, X_rp, X_fs, X_km, X_em]
labels = ['Raw', 'PCA', 'ICA', 'Random Projection', 'RFECV',
          'Kmeans', 'Expectation Maximum']

print('Collect data...')
for i, (X_input, name) in enumerate(zip(features, labels)):
    start = time.time()
    vals = cross_val_score(nnet, X_input, y, cv=5)
    cv_scores[i] = vals.mean()
    cv_stds[i] = vals.std()
    run_times[i] = time.time() - start

fig = plt.figure(figsize=(10, 10))
barwidth = 0.25

ax1 = fig.add_subplot(211)
midpoints = np.arange(cv_scores.shape[0])
ax1.bar(
    midpoints, cv_scores, yerr=cv_stds,
    color='red', width=barwidth, capsize=5
)
ax1.set_title('Performance using difference features')
ax1.set_xlabel('Methods')
ax1.set_ylabel('Scores')
ax1.set_xticks(midpoints)
ax1.set_xticklabels(labels)

ax2 = fig.add_subplot(212)
ax2.bar(
    midpoints, run_times,
    color='blue', width=barwidth
)
ax2.set_title('Time spent for training')
ax2.set_xlabel('Methods')
ax2.set_ylabel('Time')
ax2.set_xticks(midpoints)
ax2.set_xticklabels(labels)

plt.savefig(folder+'nn_performances.png', format='png')
plt.close()
print('Done...')
