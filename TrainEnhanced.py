"""
Enhanced Multi-Class Digit Classification
Combines: Raw pixels + Hand-crafted features + Deeper network
"""

import numpy as np
import pickle
from FeatureExtractor import extract_enhanced_features, extract_enhanced_features_fit, extract_features_batch
from DeepNeuralNetwork import DeepMultiClassClassifier


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


def main():
    np.random.seed(42)
    train_images, train_labels, test_images, test_labels = load_data()
    print("\nExtracting enhanced features from training images...")
    X_train_full, norm_stats = extract_enhanced_features_fit(train_images)
    print(f"  Training feature shape: {X_train_full.shape}")
    print(f"  Normalization stats computed for {len(norm_stats)} features")
    print("\nExtracting enhanced features from test images...")
    X_test = extract_enhanced_features(test_images, norm_stats=norm_stats)
    print(f"  Test feature shape: {X_test.shape}")
    X_train, y_train, X_val, y_val = create_validation_split(X_train_full, train_labels, val_ratio=0.15)
    print(f"\nAfter validation split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    input_dim = X_train.shape[1]
    print(f"  Input dimension: {input_dim}")
    print("\n" + "=" * 60)
    print("Training Deep One-vs-All Classifier")
    print("=" * 60)
    print(f"Architecture: {input_dim} -> 128 (tanh) -> 32 (tanh) -> 1")

    classifier = DeepMultiClassClassifier(
        input_dim=input_dim,
        hidden1=128,
        hidden2=32
    )

    # Train with early stopping and weight decay
    classifier.train(
        X_train, y_train,
        X_val=X_val,
        y_val=y_val,
        lr=0.005,
        iters=500000,
        lam=0.0001
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    train_acc, train_conf = classifier.evaluate(X_train, y_train)
    print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
    val_acc, val_conf = classifier.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    test_acc, test_conf = classifier.evaluate(X_test, test_labels)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    classifier.print_results(test_conf)

    # Plot decision boundaries
    # Feature layout: 0-255 = raw pixels, 256-267 = handcrafted, 268-283 = pooled
    print("\nGenerating decision boundary plots...")

    # Intensity (256) vs Vertical Symmetry (257)
    classifier.plot_decision_boundaries(
        X_train, y_train,
        feature_indices=(256, 257),
        filename="deep_boundary_intensity_vsym.png"
    )

    # Intensity (256) vs Horizontal Symmetry (258)
    classifier.plot_decision_boundaries(
        X_train, y_train,
        feature_indices=(256, 258),
        filename="deep_boundary_intensity_hsym.png"
    )

    # Edge Density (263) vs Vertical Balance (264)
    classifier.plot_decision_boundaries(
        X_train, y_train,
        feature_indices=(263, 264),
        filename="deep_boundary_edge_balance.png"
    )

    # Save model and normalization stats for later use
    model_data = {
        "classifier": classifier,
        "norm_stats": norm_stats
    }
    with open("digit_classifier_enhanced.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    print("\nModel and normalization stats saved to digit_classifier_enhanced.pkl")


if __name__ == "__main__":
    main()