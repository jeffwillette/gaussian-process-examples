import numpy as np
from matplotlib import pyplot as plt

def kernel(a, b, param):
    # works because (a - b)(a - b) = a^2 + b^2 - 2ab
    # the first two terms addition shape is (n, 1) and (n,) which turns out to be (100, 100)
    # the last term is the outer product of (100, 1)(1, 100) which is then (100, 100)
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

# full covariance is
# |K   K_s |
# |K_s K_ss|
# K = k(xtrain, y_train)
# K_s = k(xtrain, x_test)
# K_ss = k(x_test, x_test)

TRAIN = 10
TEST = 100
TOTAL = TRAIN + TEST
plot = False

for i in range(10):
    ls = np.random.rand() / 2
    x_test = np.linspace(-5, 5, TEST).reshape(-1, 1)
    y_test = np.sin(x_test)

    x_train = np.random.rand(TRAIN).reshape(-1, 1) * -10 + 5  # U(-10, 10)
    y_train = np.sin(x_train) # + np.random.normal(size=(train_n, 1))
    
    x_all = np.concatenate((x_train, x_test), axis=0)
    K = kernel(x_all, x_all, ls)
    K_xx = K[:TRAIN, :TRAIN] # (train, train)
    K_xs = K[TRAIN:, :TRAIN] # (test, train)
    K_ss = K[TRAIN:, TRAIN:] # (test, test)

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

    log_prob = -1/2 * np.dot(y.T, alpha) - 

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

# Full Covariance Matrix, Partial Covariance Matrices
# explain why they look the way that they do

# TODO: write code to get the likelihood of some test points
#  - can consult this: http://krasserm.github.io/2018/03/19/gaussian-processes/

# TODO: calculate the marginal log likelihood of the data
# TODO: write a blog post about this and cite the post https://katbailey.github.io/post/gaussian-processes-for-dummies/
