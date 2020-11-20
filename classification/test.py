import numpy as np  # type: ignore

if __name__ == "__main__":
    A = np.random.randn(2, 2)

    w = np.random.rand(2)
    W = np.diag(w)
    Wsqrt = np.diag(np.sqrt(w))

    one = np.dot(W, A)
    two = np.dot(Wsqrt, np.dot(A, Wsqrt))

    print(f"one: {one}")
    print(f"two: {two}")

    print(f"W: {W}")
    print(f"WsWs: {np.dot(Wsqrt, Wsqrt)}")
