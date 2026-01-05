import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork


class OneVsAllClassifier:
    """
    One-vs-All multi-class classifier using 10 binary neural networks.
    Each network detects one digit (0-9) vs all others.
    """

    def __init__(self, input_dim=12, hidden_units=15, output_activation="tanh"):
        """
        Args:
            input_dim: Number of input features (default 12)
            hidden_units: Hidden units per classifier
            output_activation: Activation for training ('tanh' recommended)
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_activation = output_activation
        self.classifiers = {}  # digit -> NeuralNetwork

    def _create_binary_labels(self, y, target_digit):
        """Convert multi-class labels to binary: target_digit=+1, others=-1."""
        return np.where(y == target_digit, 1, -1)

    def train(self, X_train, y_train, X_val=None, y_val=None, iters=500000, use_weight_decay=False):
        """
        Train 10 binary classifiers using early stopping.

        Args:
            X_train: Training features (N x input_dim)
            y_train: Training labels (0-9)
            X_val: Validation features (for early stopping)
            y_val: Validation labels (for early stopping)
            iters: Max iterations per classifier
            use_weight_decay: Use L2 regularization to reduce overfitting
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if X_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val)

        for digit in range(10):
            print(f"\n=== Training classifier for digit {digit} ===")

            # Create binary labels
            y_train_binary = self._create_binary_labels(y_train, digit)

            # Create fresh neural network
            # Use "identity" during training for proper gradient calculation
            nn = NeuralNetwork(
                input_dim=self.input_dim,
                m=self.hidden_units,
                output_activation="identity",
                rand_init=True
            )

            # Train with early stopping if validation set provided
            if X_val is not None and y_val is not None:
                y_val_binary = self._create_binary_labels(y_val, digit)
                self._train_early_stopping_wd(
                    nn, X_train, y_train_binary, X_val, y_val_binary,
                    iters=iters, use_wd=use_weight_decay
                )
            else:
                if use_weight_decay:
                    nn.train_sgd_wd(X_train, y_train_binary, iters=iters)
                else:
                    nn.train_sgd(X_train, y_train_binary, iters=iters)

            self.classifiers[digit] = nn

    def _train_early_stopping_wd(self, nn, X_train, y_train, X_val, y_val,
                                  iters=500000, use_wd=True):
        """
        Custom early stopping training with optional weight decay.
        """
        best_Eval = float("inf")
        best_W1 = None
        best_W2 = None
        lr = 0.01
        N = len(X_train)
        lam = 0.01 / N if use_wd else 0.0

        for t in range(iters):
            i = np.random.randint(N)
            g1, g2 = nn.gradient(X_train[i], y_train[i])
            nn.W1 -= lr * (g1 + lam * nn.W1)
            nn.W2 -= lr * (g2 + lam * nn.W2)

            if t % 10000 == 0:
                Eval = nn.compute_Ein(X_val, y_val)
                if Eval < best_Eval:
                    best_Eval = Eval
                    best_W1 = nn.W1.copy()
                    best_W2 = nn.W2.copy()

        if best_W1 is not None:
            nn.W1 = best_W1
            nn.W2 = best_W2
        print(f"Best Eval: {best_Eval:.6f}")

    def predict_scores(self, X):
        """
        Get raw confidence scores from all 10 classifiers.

        Args:
            X: Feature matrix (N x input_dim)

        Returns:
            scores: (N x 10) matrix of confidence scores
        """
        X = np.array(X)
        N = X.shape[0]
        scores = np.zeros((N, 10))

        for digit, nn in self.classifiers.items():
            # Use identity activation for raw scores
            original_activation = nn.output_activation
            nn.output_activation = "identity"

            for i in range(N):
                output, _, _, _, _ = nn.forward(X[i])
                scores[i, digit] = output.item() if hasattr(output, 'item') else float(output)

            nn.output_activation = original_activation

        return scores

    def predict(self, X):
        """
        Predict digit class for each sample.

        Args:
            X: Feature matrix (N x input_dim)

        Returns:
            predictions: Array of predicted digits (0-9)
        """
        scores = self.predict_scores(X)
        return np.argmax(scores, axis=1)

    def evaluate(self, X, y):
        """
        Compute accuracy and confusion matrix.

        Args:
            X: Feature matrix
            y: True labels (0-9)

        Returns:
            accuracy: Float (0-1)
            confusion_matrix: 10x10 numpy array
        """
        y = np.array(y).astype(int)
        predictions = self.predict(X)

        accuracy = np.mean(predictions == y)

        # Build confusion matrix: rows=true, cols=predicted
        confusion = np.zeros((10, 10), dtype=int)
        for true_label, pred_label in zip(y, predictions):
            confusion[true_label, pred_label] += 1

        return accuracy, confusion

    def print_confusion_matrix(self, confusion):
        """Pretty print the confusion matrix."""
        print("\nConfusion Matrix (rows=true, cols=predicted):")
        print("     " + " ".join(f"{i:4d}" for i in range(10)))
        print("    " + "-" * 50)
        for i in range(10):
            row = " ".join(f"{confusion[i, j]:4d}" for j in range(10))
            print(f"{i:2d} | {row}")

    def print_per_class_accuracy(self, confusion):
        """Print accuracy for each digit class."""
        print("\nPer-Digit Accuracy:")
        for digit in range(10):
            total = np.sum(confusion[digit, :])
            correct = confusion[digit, digit]
            if total > 0:
                acc = correct / total * 100
                print(f"  Digit {digit}: {acc:5.1f}% ({correct:3d}/{total:3d})")
            else:
                print(f"  Digit {digit}: N/A (no samples)")

    def plot_decision_boundaries(self, X, y, feature_indices=(0, 1), resolution=200,
                                  filename="multiclass_decision_boundary.png"):
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

        feature_names = ['Intensity', 'V-Symmetry', 'H-Symmetry',
                        'Q1', 'Q2', 'Q3', 'Q4',
                        'Edge Density', 'V-Balance', 'CoM-X', 'CoM-Y', 'Hole Proxy']
        plt.xlabel(feature_names[f1])
        plt.ylabel(feature_names[f2])
        plt.title(f'Multi-Class Decision Boundaries ({feature_names[f1]} vs {feature_names[f2]})')
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"\nDecision boundary plot saved to: {filename}")
