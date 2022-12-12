import numpy as np
import time
import contextlib


def swap_for_loop(twins):
    temp = twins.copy()
    sites = np.random.binomial(1, 0.5, size=(2, 10))
    temp[np.where(sites == 1)] = 1 - temp[np.where(sites == 1)]
    return temp


def swap_for_nparray(twins):
    temp = twins.copy()
    sites = np.random.binomial(1, 0.5, size=temp.shape)
    temp[np.where(sites == 1)] = 1 - temp[np.where(sites == 1)]
    return temp


a = np.random.randint(0, 2, size=(10, 2, 10))

time_start = time.time()
for i in range(a.shape[0]):
    swap_for_loop(a[i])
print(time.time() - time_start)


time_start = time.time()
swap_for_nparray(a)
print(time.time() - time_start)


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


def generate_dataset(size, scale=3):
    object_weights = np.random.randint(1, 100, size=size)
    object_values = np.random.randint(1, 20, size=size)
    object_constraint = 100 * size // scale

    dataset = {}
    dataset['weights'] = object_weights
    dataset['values'] = object_values
    dataset['constraint'] = object_constraint

    return dataset


dataset = generate_dataset(20, scale=3)
print(dataset)
print(
    knapsack_answer(
        dataset['weights'],
        dataset['values'],
        dataset['constraint'],
        20
    )
)


@contextlib.contextmanager
def local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


with local_seed(42):
    print(np.random.randint(0, 42))
print(np.random.randint(0, 42))
