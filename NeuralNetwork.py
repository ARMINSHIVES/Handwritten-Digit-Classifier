import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_dim=2, m=10, output_activation="identity", rand_init=True):
        self.input_dim = input_dim
        self.m = m
        self.output_activation = output_activation

        if rand_init:
            self.W1 = np.random.uniform(-0.1, 0.1, (input_dim + 1, m))
            self.W2 = np.random.uniform(-0.1, 0.1, (m + 1, 1))
        else:
            self.W1 = np.full((input_dim + 1, m), 0.15)
            self.W2 = np.full((m + 1, 1), 0.15)

    def theta(self, s):
        if self.output_activation == "identity":
            return s
        if self.output_activation == "tanh":
            return np.tanh(s)
        if self.output_activation == "sign":
            return np.sign(s)
        raise ValueError("Invalid output activation")

    def theta_prime(self, s):
        if self.output_activation == "identity":
            return np.ones_like(s)
        if self.output_activation == "tanh":
            return 1 - np.tanh(s) ** 2
        if self.output_activation == "sign":
            return np.zeros_like(s)
        raise ValueError("Invalid output activation")

    def _forward_batch(self, X):
        N = X.shape[0]
        X0 = np.hstack([np.ones((N,1)), X])
        S1 = X0 @ self.W1
        H1 = np.tanh(S1)
        X1 = np.hstack([np.ones((N,1)), H1])
        S2 = X1 @ self.W2
        return X0, S1, H1, X1, S2

    def _gradient_batch(self, X, y):
        N = X.shape[0]
        X0, S1, H1, X1, S2 = self._forward_batch(X)
        Y_hat = S2.flatten()
        Delta2 = 0.5 * (Y_hat - y)[:, None]
        Delta1 = (Delta2 @ self.W2[1:].T) * (1 - H1**2)
        grad_W2 = X1.T @ Delta2 / N
        grad_W1 = X0.T @ Delta1 / N
        return grad_W1, grad_W2

    def forward(self, x):
        x0 = np.concatenate([[1.0], x])
        s1 = x0 @ self.W1
        x1_vals = np.tanh(s1)
        x1 = np.concatenate(([1.0], x1_vals))
        s2 = x1 @ self.W2
        x2 = self.theta(s2)
        return x2, s1, x1, s2, x0

    def gradient(self, x, y):
        x2, s1, x1, s2, x0 = self.forward(x)
        delta2 = 0.5 * (x2 - y) * self.theta_prime(s2)
        theta_prime_hidden = 1 - np.tanh(s1)**2
        W2_no_bias = self.W2[1:, :].flatten()
        delta1 = theta_prime_hidden * (W2_no_bias * delta2)
        grad_W2 = np.outer(x1, delta2)
        grad_W1 = np.outer(x0, delta1)
        return grad_W1, grad_W2

    def train(self, X, y, lr=0.01, epochs=2000000):
        N = len(X)
        for ep in range(epochs):
            G1, G2 = self._gradient_batch(np.array(X), np.array(y))
            self.W1 -= lr * G1
            self.W2 -= lr * G2
            if ep % 10 == 0:
                Ein = self.compute_Ein(X, y)
                print(f"Epoch {ep}, Ein = {Ein}")

    def Ein(self, x, y):
        h, _, _, _, _ = self.forward(x)
        return 0.5 * (h - y)**2
    
    def compute_Ein(self, X, y):
        X = np.array(X)
        y = np.array(y)
        _, _, _, _, S2 = self._forward_batch(X)
        Y_hat = S2.flatten()
        return np.mean(0.5 * (Y_hat - y)**2)

    def numerical_gradient(self, x, y, eps=1e-4):
        num_grad_W1 = np.zeros_like(self.W1)
        num_grad_W2 = np.zeros_like(self.W2)
        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):
                old = self.W1[i, j]
                self.W1[i, j] = old + eps
                e_plus = self.Ein(x, y)
                self.W1[i, j] = old - eps
                e_minus = self.Ein(x, y)
                num_grad_W1[i, j] = (e_plus - e_minus) / (2 * eps)
                self.W1[i, j] = old
        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                old = self.W2[i, j]
                self.W2[i, j] = old + eps
                e_plus = self.Ein(x, y)
                self.W2[i, j] = old - eps
                e_minus = self.Ein(x, y)
                num_grad_W2[i, j] = (e_plus - e_minus) / (2 * eps)
                self.W2[i, j] = old
        return num_grad_W1, num_grad_W2

    def train_variable_lr(self, X, y, iters=2_000_000):
        X = np.array(X)
        y = np.array(y)
        Ein_history = []
        lr0 = 0.01
        for t in range(iters):
            G1, G2 = self._gradient_batch(X, y)
            lr = lr0 / (1 + t / 1000)
            self.W1 -= lr * G1
            self.W2 -= lr * G2
            if t % 5000 == 0:
                Ein_history.append(self.compute_Ein(X, y))
                #print(f"Iter {t}, Ein = {Ein_history[-1]}")
        print("Final Ein:", Ein_history[-1])
        return Ein_history

    def train_sgd(self, X, y, iters=20_000_000):
        X = np.array(X)
        y = np.array(y)
        N = len(X)
        lr = 0.01
        Ein_history = []
        for t in range(iters):
            i = np.random.randint(N)
            g1, g2 = self.gradient(X[i], y[i])
            self.W1 -= lr * g1
            self.W2 -= lr * g2
            if t % 100000 == 0:
                Ein_history.append(self.compute_Ein(X, y))
                #print(f"Iter {t}, Ein = {Ein_history[-1]}")
        print("Final Ein:", Ein_history[-1])
        return Ein_history

    def train_variable_lr_wd(self, X, y, iters=2_000_000):
        X = np.array(X)
        y = np.array(y)
        N = len(X)
        lam = 0.01 / N
        Ein_history = []
        lr0 = 0.01
        for t in range(iters):
            G1, G2 = self._gradient_batch(X, y)
            lr = lr0 / (1 + t / 1000)
            self.W1 -= lr * (G1 + lam * self.W1)
            self.W2 -= lr * (G2 + lam * self.W2)
            if t % 5000 == 0:
                Ein_history.append(self.compute_Ein(X, y))
                #print(f"Iter {t}, Ein = {Ein_history[-1]}")
        print("Final Ein:", Ein_history[-1])
        return Ein_history

    def train_sgd_wd(self, X, y, iters=20_000_000):
        X = np.array(X)
        y = np.array(y)
        N = len(X)
        lam = 0.01 / N
        lr = 0.01
        Ein_history = []
        for t in range(iters):
            i = np.random.randint(N)
            g1, g2 = self.gradient(X[i], y[i])
            self.W1 -= lr * (g1 + lam * self.W1)
            self.W2 -= lr * (g2 + lam * self.W2)
            if t % 100000 == 0:
                Ein_history.append(self.compute_Ein(X, y))
                #print(f"Iter {t}, Ein = {Ein_history[-1]}")
        print("Final Ein:", Ein_history[-1])
        return Ein_history

    def train_early_stopping(self, X_train, y_train, X_val, y_val, iters=500000):
        best_Eval = float("inf")
        best_W1 = None
        best_W2 = None
        lr = 0.01
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        for t in range(iters):
            i = np.random.randint(len(X_train))
            g1, g2 = self.gradient(X_train[i], y_train[i])
            self.W1 -= lr * g1
            self.W2 -= lr * g2
            if t % 10000 == 0:
                Eval = self.compute_Ein(X_val, y_val)
                #print(f"Iter {t}, Eval = {Eval}")
                if Eval < best_Eval:
                    best_Eval = Eval
                    best_W1 = self.W1.copy()
                    best_W2 = self.W2.copy()
        self.W1 = best_W1
        self.W2 = best_W2
        print("Best Ein:", best_Eval)

    def plot_boundary(self, X, y, resolution=300, name="decision_boundary.png"):
        xs = np.linspace(-1.2, 1.2, resolution)
        ys = np.linspace(-1.2, 1.2, resolution)
        X_grid = np.array([[x_val, y_val] for y_val in ys for x_val in xs])
        original_activation = self.output_activation
        self.output_activation = "identity"
        Z = np.array([self.forward(pt)[0] for pt in X_grid]).reshape(len(ys), len(xs))
        self.output_activation = original_activation
        plt.contour(xs, ys, Z, levels=[0], colors='k')
        plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=10, edgecolors='k')
        plt.xlabel("Intensity")
        plt.ylabel("Symmetry")
        plt.title("Decision Boundary")
        plt.savefig(name)
        plt.close()

    def compute_feature_importance(self, X=None, y=None, method='weight'):
        """
        Compute feature importance for this neural network.

        Args:
            X: Feature matrix (required for 'gradient' method)
            y: Labels (required for 'gradient' method)
            method: 'weight' or 'gradient'
                - 'weight': Sum absolute weights from input to hidden layer (fast)
                - 'gradient': Average gradient magnitude per input feature (accurate)

        Returns:
            importance: Array of shape (input_dim,) with importance scores
        """
        if method == 'weight':
            # W1 shape: (input_dim + 1, m) - row 0 is bias
            # Sum absolute weights connecting each input to all hidden units
            importance = np.abs(self.W1[1:, :]).sum(axis=1)
            return importance / (np.sum(importance) + 1e-8)

        elif method == 'gradient':
            if X is None:
                raise ValueError("X is required for gradient method")
            X = np.array(X)
            n_samples = min(500, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)

            importance = np.zeros(self.input_dim)
            for i in indices:
                x = X[i]
                # Forward pass
                x0 = np.concatenate([[1.0], x])
                s1 = x0 @ self.W1
                h1 = np.tanh(s1)
                # Gradient of output w.r.t. input: dout/dx
                dh1 = 1 - h1**2  # tanh derivative
                # W2[1:] excludes bias, shape (m, 1)
                grad_input = (self.W2[1:, :].flatten() * dh1) @ self.W1[1:, :].T
                importance += np.abs(grad_input)

            importance = importance / n_samples
            return importance / (np.sum(importance) + 1e-8)

        else:
            raise ValueError(f"Unknown method: {method}")
