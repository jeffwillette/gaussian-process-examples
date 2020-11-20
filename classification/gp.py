import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

TRAIN_SIZE = 100

def kernel(a, b, param):
    # works because (a - b)(a - b) = a^2 + b^2 - 2ab
    # the first two terms addition shape is (n, 1) and (n,) which turns out to be (100, 100)
    # the last term is the outer product of (100, 1)(1, 100) which is then (100, 100)
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

def get_dataset():
    return make_moons(n_samples = TRAIN_SIZE, noise=0.1)

def log_likelihood(x, y):
    return -np.log(1 + np.exp(-y * x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d2_likelihood(p):
    return -p * (1 - p)


def fit():
    ls = np.random.rand() / 2

    x, y = get_dataset()

    # plot the training points
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i in range(2):
        ax.scatter(x[np.where(y == i), 0], x[np.where(y == i), 1])
    
    # plt.show()

    # make some testing points
    print("MAKE SOME TESTING POINTS")
    
    K = kernel(x, x, ls)

    f = np.zeros(TRAIN_SIZE) # this is the initialized zero function (function because it is one point for each training point?)
    for i in range(100):
        pi = sigmoid(f)
        W = pi * (1 - pi)

        Wsqrt = np.diag(np.sqrt(W))
        L = np.linalg.cholesky(np.eye(TRAIN_SIZE) + np.dot(Wsqrt, np.dot(K, Wsqrt)))
        b = W * f + (y - pi)

        # below line 6 of algorithm 3.1 on page 46 of GP book. This is in the rightmost parenthesis
        tmp = np.dot(np.dot(Wsqrt, K), b)
        tmp = np.linalg.solve(L, tmp)
        tmp = np.linalg.solve(L.T, tmp)
        a = b - Wsqrt.dot(tmp)
        f = K.dot(a)

        obj = -0.5 * a.T.dot(f) \
            - np.log1p(np.exp(-(y * 2 - 1) * f)).sum() \
            - np.log(np.diag(L)).sum()

    exit("STOPPED HERE. This PART IS WORKING")

    # K_xx = K[:TRAIN, :TRAIN] # (train, train)
    # K_xs = K[TRAIN:, :TRAIN] # (test, train)
    # K_ss = K[TRAIN:, TRAIN:] # (test, test)

    L = np.linalg.cholesky(K + 1e-6 * np.eye(TOTAL))
    L_xx = np.linalg.cholesky(K_xx + 1e-6 * np.eye(TRAIN))
    L_ss = np.linalg.cholesky(K_ss + 1e-6 * np.eye(TEST))

    # the intuition of why the cholesky decomposition is necessary comes from the fact that we need to transform the 
    # standard normal samples into the distribution given by mu and Sigma. Therefore this is equivalent to sampling 
    # a standard normal and then transforming it by x * sigma + mu where sigma is sqrt(var). This is just the multi-dimensional
    # generalization of the same process
    sample = np.random.normal(size=(TEST, 5))
    prior = np.dot(L_ss, sample)

    # calculate mu
    # test point covariance multiplied by what is 0? (prior assumption about label)
    alpha = np.linalg.solve(K_ss, np.zeros(L_ss.shape[0]))
    print(f"alpha: {alpha.shape} L_xs: {K_xs.shape} K_ss: {K_ss.shape}")
    mu = np.dot(K_ss, alpha)

    # calculate sigma
    var = np.diag(K_ss)
    std = np.sqrt(var)

    if plot:
        plt.plot(x_test, prior)
        plt.gca().fill_between(x_test.flat, mu.squeeze()-2*std, mu.squeeze()+2*std, color="#dddddd")
        plt.plot(x_test, mu, "r--", lw=2)
        plt.axis([-5, 5, -3, 3])
        plt.title(f"lengthscale: {ls:.2f}")
        plt.savefig(f"prior/{i}-ls-{ls:.2f}.png")
        plt.cla()

        plt.imshow(K_ss, interpolation='nearest')
        plt.savefig(f"prior/cov-ls-{ls:.2f}.png")
        plt.cla()

    # calculate mu
    # covariance of the trianing points times what equals the train y?
    # then the mu is calculated by multiplying train, test covariance by the same thing 
    alpha = np.linalg.solve(K_xx, y_train)
    mu = np.dot(K_xs, alpha)

    # calculate sigma
    v = np.linalg.solve(L_xx, K_xs.T)
    var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
    std = np.sqrt(var)

    # sample 3 functions and plot them with the decomposed portion of the (test, test) covariance matrix  
    post = mu.reshape(-1, 1) + np.dot(L[TRAIN:, TRAIN:], np.random.normal(size=(TEST, 3)))

    print(f"alpha: {alpha.shape} ytrain: {y_train.shape}")

    # THIS WAS UNFINISHED
    # log_prob = -1/2 * np.dot(y.T, alpha) - 

    print(f"post: {post.shape}")
    print(f"mu: {mu.shape}, {mu.flat}")
    print(f"x test: {x_test.shape}")
    
    print(std.shape)

    if plot: 
        plt.plot(x_train, y_train, 'bs', ms=8)
        plt.plot(x_test, post)
        plt.axis([-5, 5, -3, 3])
        plt.gca().fill_between(x_test.flat, mu.squeeze()-2*std, mu.squeeze()+2*std, color="#dddddd")
        plt.plot(x_test, mu, "r--", lw=2)
        plt.title(f"posterior-ls-{ls:.2f}")
        plt.savefig(f"post/posterior-{i}-ls-{ls:.2f}.png")
        plt.cla()

        plt.imshow(K, interpolation='nearest')
        plt.savefig(f"post/cov-ls-{ls:.2f}.png")
        plt.cla()

if __name__ == "__main__":
    fit()
