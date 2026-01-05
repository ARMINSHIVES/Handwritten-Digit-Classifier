import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    A deeper neural network with 2 hidden layers for better feature learning.
    Architecture: input -> hidden1 (tanh) -> hidden2 (tanh) -> output
    """

    def __init__(self, input_dim, hidden1=128, hidden2=32, output_dim=1,
                 output_activation="identity", rand_init=True):
        """
        Args:
            input_dim: Number of input features
            hidden1: Units in first hidden layer
            hidden2: Units in second hidden layer
            output_dim: Number of outputs (1 for binary, 10 for multi-class)
            output_activation: 'identity', 'tanh', or 'softmax'
        """
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = output_dim
        self.output_activation = output_activation

        # Xavier/He initialization for better gradient flow
        if rand_init:
            scale1 = np.sqrt(2.0 / (input_dim + 1))
            scale2 = np.sqrt(2.0 / (hidden1 + 1))
            scale3 = np.sqrt(2.0 / (hidden2 + 1))

            self.W1 = np.random.randn(input_dim + 1, hidden1) * scale1
            self.W2 = np.random.randn(hidden1 + 1, hidden2) * scale2
            self.W3 = np.random.randn(hidden2 + 1, output_dim) * scale3
        else:
            self.W1 = np.full((input_dim + 1, hidden1), 0.01)
            self.W2 = np.full((hidden1 + 1, hidden2), 0.01)
            self.W3 = np.full((hidden2 + 1, output_dim), 0.01)

    def _add_bias(self, X):
        """Add bias column to input."""
        N = X.shape[0] if X.ndim > 1 else 1
        if X.ndim == 1:
            return np.concatenate([[1.0], X])
        return np.hstack([np.ones((N, 1)), X])

    def forward(self, x):
        """Single sample forward pass."""
        # Input layer
        x0 = np.concatenate([[1.0], x])

        # Hidden layer 1
        s1 = x0 @ self.W1
        h1 = np.tanh(s1)
        x1 = np.concatenate([[1.0], h1])

        # Hidden layer 2
        s2 = x1 @ self.W2
        h2 = np.tanh(s2)
        x2 = np.concatenate([[1.0], h2])

        # Output layer
        s3 = x2 @ self.W3
        if self.output_activation == "identity":
            out = s3
        elif self.output_activation == "tanh":
            out = np.tanh(s3)
        elif self.output_activation == "softmax":
            exp_s = np.exp(s3 - np.max(s3))
            out = exp_s / np.sum(exp_s)
        else:
            out = s3

        return out, (x0, s1, h1, x1, s2, h2, x2, s3)

    def forward_batch(self, X):
        """Batch forward pass."""
        N = X.shape[0]

        # Input layer
        X0 = self._add_bias(X)

        # Hidden layer 1
        S1 = X0 @ self.W1
        H1 = np.tanh(S1)
        X1 = self._add_bias(H1)

        # Hidden layer 2
        S2 = X1 @ self.W2
        H2 = np.tanh(S2)
        X2 = self._add_bias(H2)

        # Output layer
        S3 = X2 @ self.W3
        if self.output_activation == "softmax":
            exp_s = np.exp(S3 - np.max(S3, axis=1, keepdims=True))
            Out = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        else:
            Out = S3

        return Out, (X0, S1, H1, X1, S2, H2, X2, S3)

    def gradient(self, x, y):
        """Compute gradients for single sample using backpropagation."""
        out, (x0, s1, h1, x1, s2, h2, x2, s3) = self.forward(x)

        # Output layer gradient
        if self.output_activation == "identity":
            delta3 = (out - y)
        else:
            delta3 = (out - y)

        # Backprop to hidden2
        dh2 = (1 - h2 ** 2)
        delta2 = (self.W3[1:, :] @ delta3) * dh2

        # Backprop to hidden1
        dh1 = (1 - h1 ** 2)
        delta1 = (self.W2[1:, :] @ delta2) * dh1

        # Compute gradients
        grad_W3 = np.outer(x2, delta3)
        grad_W2 = np.outer(x1, delta2)
        grad_W1 = np.outer(x0, delta1)

        return grad_W1, grad_W2, grad_W3

    def compute_loss(self, X, y):
        """Compute mean squared error loss."""
        X = np.array(X)
        y = np.array(y)
        Out, _ = self.forward_batch(X)
        if self.output_dim == 1:
            Out = Out.flatten()
        return np.mean(0.5 * (Out - y) ** 2)

    def train_sgd(self, X, y, lr=0.01, iters=1000000, lam=0.0, verbose=True):
        """
        Train with SGD and optional weight decay.

        Args:
            X: Training features
            y: Training labels
            lr: Learning rate
            iters: Number of iterations
            lam: L2 regularization strength
            verbose: Print progress
        """
        X = np.array(X)
        y = np.array(y)
        N = len(X)
        loss_history = []

        for t in range(iters):
            i = np.random.randint(N)
            g1, g2, g3 = self.gradient(X[i], y[i])

            # Update with weight decay
            self.W1 -= lr * (g1 + lam * self.W1)
            self.W2 -= lr * (g2 + lam * self.W2)
            self.W3 -= lr * (g3 + lam * self.W3)

            if t % 50000 == 0:
                loss = self.compute_loss(X, y)
                loss_history.append(loss)
                if verbose:
                    print(f"Iter {t}, Loss = {loss:.6f}")

        return loss_history

    def train_early_stopping(self, X_train, y_train, X_val, y_val,
                              lr=0.01, iters=1000000, lam=0.0, patience=50):
        """
        Train with early stopping based on validation loss.

        Args:
            patience: Number of eval cycles without improvement before stopping
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        N = len(X_train)

        best_val_loss = float("inf")
        best_W1, best_W2, best_W3 = None, None, None
        no_improve_count = 0

        for t in range(iters):
            i = np.random.randint(N)
            g1, g2, g3 = self.gradient(X_train[i], y_train[i])

            self.W1 -= lr * (g1 + lam * self.W1)
            self.W2 -= lr * (g2 + lam * self.W2)
            self.W3 -= lr * (g3 + lam * self.W3)

            if t % 10000 == 0:
                val_loss = self.compute_loss(X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_W1 = self.W1.copy()
                    best_W2 = self.W2.copy()
                    best_W3 = self.W3.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    print(f"Early stopping at iter {t}, best val loss: {best_val_loss:.6f}")
                    break

        if best_W1 is not None:
            self.W1 = best_W1
            self.W2 = best_W2
            self.W3 = best_W3

        print(f"Best validation loss: {best_val_loss:.6f}")
        return best_val_loss

    def predict(self, X):
        """Get raw output for batch."""
        Out, _ = self.forward_batch(np.array(X))
        return Out.flatten() if self.output_dim == 1 else Out

    def compute_feature_importance(self, X=None, y=None, method='weight'):
        """
        Compute feature importance for this deep neural network.

        Args:
            X: Feature matrix (required for 'gradient' method)
            y: Labels (required for 'gradient' method)
            method: 'weight' or 'gradient'
                - 'weight': Sum absolute weights from input to first hidden layer (fast)
                - 'gradient': Average gradient magnitude per input feature (accurate)

        Returns:
            importance: Array of shape (input_dim,) with importance scores
        """
        if method == 'weight':
            # W1 shape: (input_dim + 1, hidden1) - row 0 is bias
            # Sum absolute weights connecting each input to all first hidden units
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
                out, (x0, s1, h1, x1, s2, h2, x2, s3) = self.forward(x)
                # Backprop gradient to input
                dh2 = 1 - h2**2
                dh1 = 1 - h1**2
                # W3[1:] shape: (hidden2, output_dim)
                delta2 = self.W3[1:, :].flatten() * dh2
                # W2[1:] shape: (hidden1, hidden2)
                delta1 = (self.W2[1:, :] @ delta2) * dh1
                # W1[1:] shape: (input_dim, hidden1)
                grad_input = self.W1[1:, :] @ delta1
                importance += np.abs(grad_input)

            importance = importance / n_samples
            return importance / (np.sum(importance) + 1e-8)

        else:
            raise ValueError(f"Unknown method: {method}")


class DeepMultiClassClassifier:
    """
    One-vs-All classifier using deep neural networks.
    """

    def __init__(self, input_dim, hidden1=128, hidden2=32):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.classifiers = {}

    def train(self, X_train, y_train, X_val=None, y_val=None,
              lr=0.005, iters=500000, lam=0.0001):
        """Train 10 binary classifiers."""
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        for digit in range(10):
            print(f"\n=== Training classifier for digit {digit} ===")

            # Binary labels
            y_binary = np.where(y_train == digit, 1, -1)

            # Create network
            nn = DeepNeuralNetwork(
                input_dim=self.input_dim,
                hidden1=self.hidden1,
                hidden2=self.hidden2,
                output_dim=1,
                output_activation="identity"
            )

            # Train
            if X_val is not None:
                y_val_binary = np.where(y_val == digit, 1, -1)
                nn.train_early_stopping(
                    X_train, y_binary, X_val, y_val_binary,
                    lr=lr, iters=iters, lam=lam, patience=30
                )
            else:
                nn.train_sgd(X_train, y_binary, lr=lr, iters=iters, lam=lam)

            self.classifiers[digit] = nn

    def predict_scores(self, X):
        """Get scores from all classifiers."""
        X = np.array(X)
        N = X.shape[0]
        scores = np.zeros((N, 10))

        for digit, nn in self.classifiers.items():
            scores[:, digit] = nn.predict(X)

        return scores

    def predict(self, X):
        """Predict digit class."""
        scores = self.predict_scores(X)
        return np.argmax(scores, axis=1)

    def evaluate(self, X, y):
        """Compute accuracy and confusion matrix."""
        y = np.array(y).astype(int)
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        confusion = np.zeros((10, 10), dtype=int)
        for true_label, pred_label in zip(y, predictions):
            confusion[true_label, pred_label] += 1

        return accuracy, confusion

    def print_results(self, confusion):
        """Print confusion matrix and per-class accuracy."""
        print("\nConfusion Matrix (rows=true, cols=predicted):")
        print("     " + " ".join(f"{i:4d}" for i in range(10)))
        print("    " + "-" * 50)
        for i in range(10):
            row = " ".join(f"{confusion[i, j]:4d}" for j in range(10))
            print(f"{i:2d} | {row}")

        print("\nPer-Digit Accuracy:")
        for digit in range(10):
            total = np.sum(confusion[digit, :])
            correct = confusion[digit, digit]
            if total > 0:
                print(f"  Digit {digit}: {correct/total*100:5.1f}% ({correct:3d}/{total:3d})")

    def plot_decision_boundaries(self, X, y, feature_indices=(0, 1), resolution=200,
                                  filename="deep_decision_boundary.png"):
        """
        Plot decision boundaries for all 10 classes using 2 selected features.

        Args:
            X: Full feature matrix (N x input_dim)
            y: True labels (0-9)
            feature_indices: Tuple of 2 feature indices to use for visualization
            resolution: Grid resolution
            filename: Output filename
        """
        X = np.array(X)
        y = np.array(y).astype(int)
        f1, f2 = feature_indices

        # Get the 2 features for plotting
        X_2d = X[:, [f1, f2]]

        # Create grid
        x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
        y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # For each grid point, create full feature vector using mean of other features
        X_mean = np.mean(X, axis=0)
        grid_points = []
        for i in range(resolution):
            for j in range(resolution):
                point = X_mean.copy()
                point[f1] = xx[i, j]
                point[f2] = yy[i, j]
                grid_points.append(point)
        grid_points = np.array(grid_points)

        # Get predictions for grid
        Z = self.predict(grid_points).reshape(resolution, resolution)

        # Plot
        plt.figure(figsize=(12, 10))

        # Decision regions
        cmap = plt.cm.get_cmap('tab10', 10)
        plt.contourf(xx, yy, Z, levels=np.arange(-0.5, 10.5, 1), cmap=cmap, alpha=0.3)

        # Decision boundaries
        plt.contour(xx, yy, Z, levels=np.arange(-0.5, 10.5, 1), colors='k', linewidths=0.5)

        # Plot data points
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap,
                             s=15, edgecolors='k', linewidths=0.3, alpha=0.7)

        # Colorbar
        cbar = plt.colorbar(scatter, ticks=range(10))
        cbar.set_label('Digit')

        # Feature names for enhanced features (284 total)
        # First 256 are raw pixels, then 12 handcrafted, then 16 pooled
        if f1 < 256:
            name1 = f"Pixel {f1}"
        elif f1 < 268:
            handcrafted_names = ['Intensity', 'V-Symmetry', 'H-Symmetry',
                                'Q1', 'Q2', 'Q3', 'Q4',
                                'Edge Density', 'V-Balance', 'CoM-X', 'CoM-Y', 'Hole Proxy']
            name1 = handcrafted_names[f1 - 256]
        else:
            name1 = f"Pooled {f1 - 268}"

        if f2 < 256:
            name2 = f"Pixel {f2}"
        elif f2 < 268:
            handcrafted_names = ['Intensity', 'V-Symmetry', 'H-Symmetry',
                                'Q1', 'Q2', 'Q3', 'Q4',
                                'Edge Density', 'V-Balance', 'CoM-X', 'CoM-Y', 'Hole Proxy']
            name2 = handcrafted_names[f2 - 256]
        else:
            name2 = f"Pooled {f2 - 268}"

        plt.xlabel(name1)
        plt.ylabel(name2)
        plt.title(f'Deep Multi-Class Decision Boundaries ({name1} vs {name2})')
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"\nDecision boundary plot saved to: {filename}")
