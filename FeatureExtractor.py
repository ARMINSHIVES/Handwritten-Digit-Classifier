import numpy as np


def compute_intensity(image):
    """Feature 1: Mean pixel value (overall darkness)."""
    return np.mean(image)


def compute_vertical_symmetry(image):
    """Feature 2: Left-right mirror comparison (negative MSE)."""
    im = image.reshape(16, 16)
    left = im[:, :8]
    right = np.fliplr(im[:, 8:])
    return -np.mean((left - right) ** 2)


def compute_horizontal_symmetry(image):
    """Feature 3: Top-bottom mirror comparison (negative MSE)."""
    im = image.reshape(16, 16)
    top = im[:8, :]
    bottom = np.flipud(im[8:, :])
    return -np.mean((top - bottom) ** 2)


def compute_quadrant_intensities(image):
    """Features 4-7: Mean intensity in each 8x8 quadrant."""
    im = image.reshape(16, 16)
    q1 = np.mean(im[:8, :8])    # top-left
    q2 = np.mean(im[:8, 8:])    # top-right
    q3 = np.mean(im[8:, :8])    # bottom-left
    q4 = np.mean(im[8:, 8:])    # bottom-right
    return q1, q2, q3, q4


def compute_edge_density(image):
    """Feature 8: Sum of pixel transitions (captures stroke complexity)."""
    im = image.reshape(16, 16)
    h_edges = np.abs(im[:, 1:] - im[:, :-1])
    v_edges = np.abs(im[1:, :] - im[:-1, :])
    return np.mean(h_edges) + np.mean(v_edges)


def compute_vertical_balance(image):
    """Feature 9: Top vs bottom intensity ratio."""
    im = image.reshape(16, 16)
    top_intensity = np.mean(im[:8, :])
    bottom_intensity = np.mean(im[8:, :])
    return (top_intensity - bottom_intensity) / (top_intensity + bottom_intensity + 1e-8)


def compute_center_of_mass(image):
    """Features 10-11: Weighted centroid (x, y) of dark pixels."""
    im = image.reshape(16, 16)
    # In this dataset, higher values = darker pixels (ink)
    weights = im - np.min(im)  # Shift to positive
    total = np.sum(weights) + 1e-8
    y_indices, x_indices = np.mgrid[0:16, 0:16]
    cx = np.sum(x_indices * weights) / total
    cy = np.sum(y_indices * weights) / total
    # Normalize to [-1, 1] range
    return (cx - 7.5) / 7.5, (cy - 7.5) / 7.5


def compute_hole_proxy(image):
    """Feature 12: Center vs edge intensity ratio (digits with holes have different center)."""
    im = image.reshape(16, 16)
    # Center region (6x6 in the middle)
    center = im[5:11, 5:11]
    # Edge region (border pixels, 2 pixels thick)
    edge_mask = np.ones((16, 16), dtype=bool)
    edge_mask[2:14, 2:14] = False
    edge = im[edge_mask]
    center_mean = np.mean(center)
    edge_mean = np.mean(edge)
    return (center_mean - edge_mean) / (center_mean + edge_mean + 1e-8)


def extract_all_features(image):
    """Extract all 12 features from a single 256-element image array."""
    intensity = compute_intensity(image)
    v_sym = compute_vertical_symmetry(image)
    h_sym = compute_horizontal_symmetry(image)
    q1, q2, q3, q4 = compute_quadrant_intensities(image)
    edge_density = compute_edge_density(image)
    v_balance = compute_vertical_balance(image)
    cx, cy = compute_center_of_mass(image)
    hole_proxy = compute_hole_proxy(image)

    return np.array([
        intensity,      # 1
        v_sym,          # 2
        h_sym,          # 3
        q1, q2, q3, q4, # 4-7
        edge_density,   # 8
        v_balance,      # 9
        cx, cy,         # 10-11
        hole_proxy      # 12
    ])


def normalize_features(X):
    """Normalize each feature column to [-1, 1] range using min-max scaling."""
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[1]):
        col = X[:, i]
        x_min, x_max = np.min(col), np.max(col)
        center = (x_max + x_min) / 2
        scale = (x_max - x_min) / 2 + 1e-8
        X_normalized[:, i] = (col - center) / scale
    return X_normalized


def normalize_features_fit(X):
    """Compute normalization statistics from training data.

    Returns:
        List of (center, scale) tuples for each feature column.
    """
    stats = []
    for i in range(X.shape[1]):
        col = X[:, i]
        x_min, x_max = np.min(col), np.max(col)
        center = (x_max + x_min) / 2
        scale = (x_max - x_min) / 2 + 1e-8
        stats.append((center, scale))
    return stats


def normalize_features_transform(X, stats):
    """Apply precomputed normalization statistics to new data.

    Args:
        X: Feature matrix to normalize
        stats: List of (center, scale) tuples from normalize_features_fit()

    Returns:
        Normalized feature matrix
    """
    X_normalized = np.zeros_like(X)
    for i, (center, scale) in enumerate(stats):
        X_normalized[:, i] = (X[:, i] - center) / scale
    return X_normalized


def extract_features_batch(images):
    """Extract and normalize features for all images.

    Args:
        images: (N, 256) array of flattened 16x16 images

    Returns:
        (N, 12) array of normalized features
    """
    features = np.array([extract_all_features(img) for img in images])
    return normalize_features(features)


def extract_combined_features(images, use_raw_pixels=True, use_handcrafted=True,
                               use_pooled=True):
    """
    Extract combined feature set for maximum accuracy.

    Args:
        images: (N, 256) array of flattened 16x16 images
        use_raw_pixels: Include normalized raw pixels (256 features)
        use_handcrafted: Include hand-crafted features (12 features)
        use_pooled: Include 4x4 pooled pixels (16 features)

    Returns:
        (N, total_features) normalized feature array
        total_features = 256 + 12 + 16 = 284 (if all enabled)
    """
    images = np.array(images)
    feature_sets = []

    # 1. Raw pixels (256 features) - normalized
    if use_raw_pixels:
        raw_normalized = normalize_features(images)
        feature_sets.append(raw_normalized)

    # 2. Hand-crafted features (12 features)
    if use_handcrafted:
        handcrafted = np.array([extract_all_features(img) for img in images])
        handcrafted_normalized = normalize_features(handcrafted)
        feature_sets.append(handcrafted_normalized)

    # 3. Pooled features (16 features) - 4x4 average pooling
    if use_pooled:
        pooled = []
        for img in images:
            im = img.reshape(16, 16)
            # 4x4 average pooling -> 4x4 = 16 values
            pooled_img = []
            for i in range(4):
                for j in range(4):
                    pooled_img.append(np.mean(im[i*4:(i+1)*4, j*4:(j+1)*4]))
            pooled.append(pooled_img)
        pooled = np.array(pooled)
        pooled_normalized = normalize_features(pooled)
        feature_sets.append(pooled_normalized)

    # Concatenate all feature sets
    return np.hstack(feature_sets)


def extract_enhanced_features(images, norm_stats=None):
    """
    Extract enhanced feature set optimized for digit recognition.
    Includes: raw pixels (256) + hand-crafted (12) + pooled (16) = 284 features

    Args:
        images: (N, 256) array of flattened 16x16 images
        norm_stats: If provided, use these stats for normalization (prediction mode).
                   If None, compute stats from the batch (training mode).

    Returns:
        If norm_stats is None: normalized features (training mode)
        If norm_stats is provided: normalized features using saved stats (prediction mode)
    """
    images = np.array(images)
    feature_sets = []

    # 1. Raw pixels (256 features)
    feature_sets.append(images)

    # 2. Hand-crafted features (12 features)
    handcrafted = np.array([extract_all_features(img) for img in images])
    feature_sets.append(handcrafted)

    # 3. Pooled features (16 features) - 4x4 average pooling
    pooled = []
    for img in images:
        im = img.reshape(16, 16)
        pooled_img = []
        for i in range(4):
            for j in range(4):
                pooled_img.append(np.mean(im[i*4:(i+1)*4, j*4:(j+1)*4]))
        pooled.append(pooled_img)
    pooled = np.array(pooled)
    feature_sets.append(pooled)

    # Concatenate all feature sets
    combined = np.hstack(feature_sets)

    # Normalize using provided stats or compute from batch
    if norm_stats is not None:
        return normalize_features_transform(combined, norm_stats)
    else:
        return normalize_features(combined)


def extract_enhanced_features_fit(images):
    """
    Extract enhanced features and compute normalization statistics.
    Use this during training to get both features and stats to save.

    Args:
        images: (N, 256) array of flattened 16x16 images

    Returns:
        features: Normalized feature matrix
        norm_stats: Normalization statistics to save for prediction
    """
    images = np.array(images)
    feature_sets = []

    # 1. Raw pixels (256 features)
    feature_sets.append(images)

    # 2. Hand-crafted features (12 features)
    handcrafted = np.array([extract_all_features(img) for img in images])
    feature_sets.append(handcrafted)

    # 3. Pooled features (16 features) - 4x4 average pooling
    pooled = []
    for img in images:
        im = img.reshape(16, 16)
        pooled_img = []
        for i in range(4):
            for j in range(4):
                pooled_img.append(np.mean(im[i*4:(i+1)*4, j*4:(j+1)*4]))
        pooled.append(pooled_img)
    pooled = np.array(pooled)
    feature_sets.append(pooled)

    # Concatenate all feature sets
    combined = np.hstack(feature_sets)

    # Compute normalization stats and normalize
    norm_stats = normalize_features_fit(combined)
    features = normalize_features_transform(combined, norm_stats)

    return features, norm_stats
