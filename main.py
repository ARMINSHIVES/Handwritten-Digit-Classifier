"""
Unified Neural Network Training Script for Digit Classification

Usage:
    python main.py --network deep --features enhanced --analyze
    python main.py --network shallow --features basic --hidden 15
    python main.py --help
"""

import argparse
import numpy as np
import pickle
from FeatureExtractor import (
    extract_features_batch,
    extract_enhanced_features_fit,
    extract_enhanced_features
)
from NeuralNetwork import NeuralNetwork
from DeepNeuralNetwork import DeepNeuralNetwork, DeepMultiClassClassifier
from MultiClassClassifier import OneVsAllClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train digit classifier with configurable neural network architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --network deep --features enhanced --analyze
  python main.py --network shallow --features basic --hidden 15
  python main.py --network deep --hidden 64,16 --lr 0.01 --iters 300000
        """
    )
    parser.add_argument('--network', choices=['shallow', 'deep'], default='deep',
                        help='Network type: shallow (2-layer) or deep (3-layer)')
    parser.add_argument('--features', choices=['basic', 'enhanced'], default='enhanced',
                        help='Feature set: basic (12 handcrafted) or enhanced (284 combined)')
    parser.add_argument('--hidden', type=str, default=None,
                        help='Hidden layer sizes (e.g., "15" for shallow, "128,32" for deep)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.01 for shallow, 0.005 for deep)')
    parser.add_argument('--iters', type=int, default=None,
                        help='Training iterations per classifier (default: 2M shallow, 500K deep)')
    parser.add_argument('--lam', type=float, default=None,
                        help='L2 regularization strength (default: 0.01/N shallow, 0.0001 deep)')
    parser.add_argument('--output', type=str, default='digit_classifier.pkl',
                        help='Model output filename')
    parser.add_argument('--analyze', action='store_true',
                        help='Run feature importance analysis after training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating decision boundary plots')
    return parser.parse_args()


def load_data():
    """Load the ZIP digits training and test data."""
    print("Loading ZIP digit data...")
    train = np.loadtxt("ZipDigits.train")
    test = np.loadtxt("ZipDigits.test")

    train_labels = train[:, 0].astype(int)
    train_images = train[:, 1:]
    test_labels = test[:, 0].astype(int)
    test_images = test[:, 1:]

    print(f"  Training samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")

    return train_images, train_labels, test_images, test_labels


def create_validation_split(X, y, val_ratio=0.15):
    """Split data into training and validation sets."""
    N = len(X)
    val_size = int(N * val_ratio)
    indices = np.random.permutation(N)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def extract_features(images, feature_type, fit=False, norm_stats=None):
    """
    Extract features based on type.

    Args:
        images: Raw image data
        feature_type: 'basic' (12 features) or 'enhanced' (284 features)
        fit: If True, compute and return normalization stats
        norm_stats: Pre-computed stats for transform (prediction mode)

    Returns:
        features: Extracted feature matrix
        norm_stats: (only if fit=True) Normalization statistics
    """
    if feature_type == 'basic':
        features = extract_features_batch(images)
        return (features, None) if fit else features
    else:
        if fit:
            return extract_enhanced_features_fit(images)
        else:
            return extract_enhanced_features(images, norm_stats=norm_stats)


def create_classifier(network_type, input_dim, hidden_sizes):
    """
    Create the appropriate classifier.

    Args:
        network_type: 'shallow' or 'deep'
        input_dim: Number of input features
        hidden_sizes: List of hidden layer sizes

    Returns:
        classifier: OneVsAllClassifier or DeepMultiClassClassifier
    """
    if network_type == 'shallow':
        hidden = hidden_sizes[0] if hidden_sizes else 15
        print(f"\nCreating shallow classifier: {input_dim} -> {hidden} -> 1")
        return OneVsAllClassifier(
            input_dim=input_dim,
            hidden_units=hidden,
            output_activation="tanh"
        )
    else:
        h1 = hidden_sizes[0] if len(hidden_sizes) > 0 else 128
        h2 = hidden_sizes[1] if len(hidden_sizes) > 1 else 32
        print(f"\nCreating deep classifier: {input_dim} -> {h1} -> {h2} -> 1")
        return DeepMultiClassClassifier(
            input_dim=input_dim,
            hidden1=h1,
            hidden2=h2
        )


def get_feature_names(feature_type, input_dim):
    """Get human-readable feature names."""
    if feature_type == 'basic':
        return ['Intensity', 'V-Symmetry', 'H-Symmetry',
                'Q1', 'Q2', 'Q3', 'Q4',
                'Edge Density', 'V-Balance', 'CoM-X', 'CoM-Y', 'Hole Proxy']
    else:
        names = []
        # Raw pixels (0-255)
        for i in range(256):
            row, col = i // 16, i % 16
            names.append(f"Pixel[{row},{col}]")
        # Handcrafted (256-267)
        names.extend(['Intensity', 'V-Symmetry', 'H-Symmetry',
                     'Q1', 'Q2', 'Q3', 'Q4',
                     'Edge Density', 'V-Balance', 'CoM-X', 'CoM-Y', 'Hole Proxy'])
        # Pooled (268-283)
        for i in range(16):
            row, col = i // 4, i % 4
            names.append(f"Pooled[{row},{col}]")
        return names


def analyze_feature_importance(classifier, X, y, feature_names, method='weight'):
    """
    Analyze feature importance for the trained classifier.

    Args:
        classifier: Trained OneVsAllClassifier or DeepMultiClassClassifier
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        method: 'weight', 'gradient', or 'permutation'

    Returns:
        importance_dict: {digit: importance_array}
        overall_importance: Combined importance across all digits
    """
    print(f"\n{'='*60}")
    print(f"Feature Importance Analysis (method: {method})")
    print('='*60)

    input_dim = X.shape[1]
    importance_per_digit = {}

    for digit in range(10):
        nn = classifier.classifiers[digit]

        if method == 'weight':
            # Sum absolute weights from input layer to first hidden layer
            # W1 shape: (input_dim + 1, hidden_units) - first row is bias
            importance = np.abs(nn.W1[1:, :]).sum(axis=1)  # Skip bias row

        elif method == 'gradient':
            # Compute average gradient magnitude per feature
            importance = compute_gradient_importance(nn, X, y, digit)

        elif method == 'permutation':
            # Measure accuracy drop when each feature is shuffled
            importance = compute_permutation_importance(classifier, X, y, digit)

        # Normalize to sum to 1
        importance = importance / (np.sum(importance) + 1e-8)
        importance_per_digit[digit] = importance

    # Compute overall importance (average across digits)
    overall_importance = np.mean([importance_per_digit[d] for d in range(10)], axis=0)
    overall_importance = overall_importance / (np.sum(overall_importance) + 1e-8)

    # Print top features overall
    print("\nTop 15 Most Important Features (Overall):")
    print("-" * 50)
    top_indices = np.argsort(overall_importance)[::-1][:15]
    for rank, idx in enumerate(top_indices, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        print(f"  {rank:2d}. {name:20s} (idx {idx:3d}): {overall_importance[idx]:.4f}")

    # Print top features per digit
    print("\nTop 5 Features Per Digit:")
    print("-" * 50)
    for digit in range(10):
        top_idx = np.argsort(importance_per_digit[digit])[::-1][:5]
        features_str = ", ".join([
            f"{feature_names[i] if i < len(feature_names) else f'F{i}'}"
            for i in top_idx
        ])
        print(f"  Digit {digit}: {features_str}")

    # Feature reduction recommendation
    threshold = 0.001
    low_importance_count = np.sum(overall_importance < threshold)
    if low_importance_count > 0:
        print(f"\nRecommendation:")
        print(f"  {low_importance_count} features have importance < {threshold}")
        print(f"  Consider removing these for faster training with minimal accuracy loss.")

    return importance_per_digit, overall_importance


def compute_gradient_importance(nn, X, y, digit):
    """Compute gradient-based feature importance for a single classifier."""
    X = np.array(X)
    input_dim = X.shape[1]

    # Sample a subset for efficiency
    n_samples = min(500, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)

    importance = np.zeros(input_dim)

    for i in indices:
        x = X[i]
        # Compute gradient of output with respect to input
        out, cache = nn.forward(x)

        # For shallow network
        if hasattr(nn, 'W2') and not hasattr(nn, 'W3'):
            x0, s1, h1, x1, s2 = cache if len(cache) == 5 else (None,) * 5
            if x0 is None:
                continue
            # dout/dinput = W2[1:].T @ diag(1-tanh(s1)^2) @ W1[1:,:].T
            dh1 = 1 - np.tanh(s1)**2
            grad_input = (nn.W2[1:, :].flatten() * dh1) @ nn.W1[1:, :].T
        else:
            # For deep network
            x0, s1, h1, x1, s2, h2, x2, s3 = cache
            # Backprop gradient to input
            dh2 = 1 - h2**2
            dh1 = 1 - h1**2
            delta2 = nn.W3[1:, :].flatten() * dh2
            delta1 = (nn.W2[1:, :] @ delta2) * dh1
            grad_input = nn.W1[1:, :] @ delta1

        importance += np.abs(grad_input)

    return importance / n_samples


def compute_permutation_importance(classifier, X, y, digit):
    """Compute permutation importance for a single digit."""
    X = np.array(X)
    y = np.array(y)
    input_dim = X.shape[1]

    # Baseline accuracy for this digit
    predictions = classifier.predict(X)
    baseline_acc = np.mean((predictions == digit) == (y == digit))

    importance = np.zeros(input_dim)

    # Sample features to test (for efficiency with high-dim data)
    n_features_to_test = min(50, input_dim)
    feature_indices = np.random.choice(input_dim, n_features_to_test, replace=False)

    for idx in feature_indices:
        X_permuted = X.copy()
        X_permuted[:, idx] = np.random.permutation(X_permuted[:, idx])

        predictions = classifier.predict(X_permuted)
        permuted_acc = np.mean((predictions == digit) == (y == digit))

        importance[idx] = max(0, baseline_acc - permuted_acc)

    return importance


def plot_decision_boundaries(classifier, X, y, feature_type, args):
    """Generate decision boundary plots."""
    if args.no_plot:
        return

    print("\nGenerating decision boundary plots...")

    if feature_type == 'basic':
        # Use handcrafted feature indices
        pairs = [
            ((0, 1), "boundary_intensity_vsym.png"),
            ((0, 2), "boundary_intensity_hsym.png"),
            ((7, 8), "boundary_edge_balance.png"),
        ]
    else:
        # Enhanced features: handcrafted are at indices 256-267
        pairs = [
            ((256, 257), "boundary_intensity_vsym.png"),
            ((256, 258), "boundary_intensity_hsym.png"),
            ((263, 264), "boundary_edge_balance.png"),
        ]

    for (f1, f2), filename in pairs:
        classifier.plot_decision_boundaries(
            X, y,
            feature_indices=(f1, f2),
            filename=filename
        )


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load data
    train_images, train_labels, test_images, test_labels = load_data()

    # Set defaults based on network type
    if args.network == 'shallow':
        default_hidden = [15]
        default_lr = 0.01
        default_iters = 2000000
    else:
        default_hidden = [128, 32]
        default_lr = 0.005
        default_iters = 500000

    # Parse hidden layers
    if args.hidden:
        hidden_sizes = [int(h.strip()) for h in args.hidden.split(',')]
    else:
        hidden_sizes = default_hidden

    lr = args.lr if args.lr is not None else default_lr
    iters = args.iters if args.iters is not None else default_iters

    # Extract features
    print(f"\nExtracting {args.features} features from training images...")
    X_train_full, norm_stats = extract_features(train_images, args.features, fit=True)
    print(f"  Feature shape: {X_train_full.shape}")

    print(f"\nExtracting {args.features} features from test images...")
    X_test = extract_features(test_images, args.features, norm_stats=norm_stats)
    print(f"  Feature shape: {X_test.shape}")

    # Create validation split
    X_train, y_train, X_val, y_val = create_validation_split(
        X_train_full, train_labels, val_ratio=args.val_ratio
    )
    print(f"\nAfter validation split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")

    # Create classifier
    input_dim = X_train.shape[1]
    classifier = create_classifier(args.network, input_dim, hidden_sizes)

    # Training configuration
    print(f"\n{'='*60}")
    print("Training Configuration")
    print('='*60)
    print(f"  Network type: {args.network}")
    print(f"  Features: {args.features} ({input_dim} dimensions)")
    print(f"  Hidden layers: {hidden_sizes}")
    print(f"  Learning rate: {lr}")
    print(f"  Iterations: {iters:,}")

    # Train
    print(f"\n{'='*60}")
    print("Training")
    print('='*60)

    if args.network == 'shallow':
        use_wd = True
        classifier.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            iters=iters,
            use_weight_decay=use_wd
        )
    else:
        lam = args.lam if args.lam is not None else 0.0001
        classifier.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            lr=lr,
            iters=iters,
            lam=lam
        )

    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print('='*60)

    train_acc, train_conf = classifier.evaluate(X_train, y_train)
    print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")

    val_acc, val_conf = classifier.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    test_acc, test_conf = classifier.evaluate(X_test, test_labels)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Print detailed results
    if hasattr(classifier, 'print_results'):
        classifier.print_results(test_conf)
    else:
        classifier.print_confusion_matrix(test_conf)
        classifier.print_per_class_accuracy(test_conf)

    # Feature importance analysis
    if args.analyze:
        feature_names = get_feature_names(args.features, input_dim)
        importance_per_digit, overall_importance = analyze_feature_importance(
            classifier, X_train, y_train, feature_names, method='weight'
        )

        # Save importance data
        importance_file = args.output.replace('.pkl', '_importance.pkl')
        with open(importance_file, 'wb') as f:
            pickle.dump({
                'per_digit': importance_per_digit,
                'overall': overall_importance,
                'feature_names': feature_names
            }, f)
        print(f"\nFeature importance saved to: {importance_file}")

    # Plot decision boundaries
    plot_decision_boundaries(classifier, X_train, y_train, args.features, args)

    # Save model
    model_data = {
        'classifier': classifier,
        'norm_stats': norm_stats,
        'config': {
            'network': args.network,
            'features': args.features,
            'hidden_sizes': hidden_sizes,
            'input_dim': input_dim
        }
    }
    with open(args.output, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to: {args.output}")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print('='*60)


if __name__ == "__main__":
    main()
