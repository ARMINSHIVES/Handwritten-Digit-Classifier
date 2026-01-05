"""
Predict handwritten digits from images using the trained neural network.
Supports photos of handwritten digits - will preprocess and predict.
"""

import numpy as np
import pickle
from PIL import Image
from FeatureExtractor import extract_enhanced_features


def save_model(classifier, filename="digit_classifier.pkl"):
    """Save trained classifier to file."""
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Model saved to {filename}")


def load_model(filename="digit_classifier.pkl"):
    """Load trained classifier and normalization stats from file."""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)

    # Handle both old format (just classifier) and new format (dict with classifier + norm_stats)
    if isinstance(model_data, dict):
        classifier = model_data["classifier"]
        norm_stats = model_data.get("norm_stats", None)
        print(f"Model loaded from {filename}")
        if norm_stats:
            print(f"  Normalization stats loaded for {len(norm_stats)} features")
        return classifier, norm_stats
    else:
        # Old format: just the classifier, no norm_stats
        print(f"Model loaded from {filename} (old format, no normalization stats)")
        print("  WARNING: Predictions may be inaccurate. Please retrain the model.")
        return model_data, None


def preprocess_image(image_path, invert=True, show_preview=False):
    """
    Preprocess an image of a handwritten digit for the neural network.

    Args:
        image_path: Path to the image file (jpg, png, etc.)
        invert: If True, invert colors (use if digit is dark on light background)
        show_preview: If True, display the preprocessed image

    Returns:
        256-element numpy array (flattened 16x16 grayscale image)
    """
    img = Image.open(image_path)
    img = img.convert('L')
    img_array = np.array(img)

    if invert:
        img_array = 255 - img_array

    threshold = img_array.max() * 0.3
    digit_pixels = img_array > threshold
    rows = np.any(digit_pixels, axis=1)
    cols = np.any(digit_pixels, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        padding = 2
        rmin = max(0, rmin - padding)
        rmax = min(img_array.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(img_array.shape[1], cmax + padding)
        img_array = img_array[rmin:rmax+1, cmin:cmax+1]

    h, w = img_array.shape
    if h > w:
        pad = (h - w) // 2
        img_array = np.pad(img_array, ((0, 0), (pad, h - w - pad)), mode='constant', constant_values=0)
    elif w > h:
        pad = (w - h) // 2
        img_array = np.pad(img_array, ((pad, w - h - pad), (0, 0)), mode='constant', constant_values=0)

    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.resize((16, 16), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float64)
    img_array = (img_array / 255.0) * 2 - 1  # Scale to [-1, 1]

    if show_preview:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4))
        plt.imshow(img_array, cmap='gray')
        plt.title("Preprocessed 16x16 image")
        plt.axis('off')
        plt.show()

    return img_array.flatten()

def predict_digit(classifier, image_input, norm_stats=None, invert=True, show_preview=False):
    """
    Predict a digit from an image.

    Args:
        classifier: Trained DeepMultiClassClassifier
        image_input: Either:
            - Path to image file (str)
            - 256-element numpy array (already preprocessed)
            - 16x16 numpy array
        norm_stats: Normalization statistics from training (required for accurate predictions)
        invert: Invert colors (for dark digit on light background)
        show_preview: Show the preprocessed image

    Returns:
        predicted_digit: int (0-9)
        scores: confidence scores for each digit
    """
    # Handle different input types
    if isinstance(image_input, str):
        # Load and preprocess from file
        image = preprocess_image(image_input, invert=invert, show_preview=show_preview)
    elif isinstance(image_input, np.ndarray):
        if image_input.shape == (16, 16):
            image = image_input.flatten()
        elif image_input.shape == (256,):
            image = image_input
        else:
            raise ValueError(f"Array must be 16x16 or 256 elements, got {image_input.shape}")
    else:
        raise ValueError("Input must be a file path or numpy array")

    # Extract features using saved normalization stats
    features = extract_enhanced_features(image.reshape(1, -1), norm_stats=norm_stats)

    # Get prediction
    scores = classifier.predict_scores(features)[0]
    predicted = np.argmax(scores)

    return predicted, scores


def predict_multi_digit(classifier, image_path, norm_stats=None, invert=True):
    """
    Predict multiple digits from an image (like "67").
    Splits the image into individual digit regions.

    Args:
        classifier: Trained classifier
        image_path: Path to image with multiple digits
        norm_stats: Normalization statistics from training (required for accurate predictions)
        invert: Invert colors

    Returns:
        predicted_number: string of predicted digits
        individual_predictions: list of (digit, scores) tuples
    """
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    if invert:
        img_array = 255 - img_array

    # Threshold
    threshold = img_array.max() * 0.3
    binary = img_array > threshold

    # Find connected components (simple column-based segmentation)
    cols_with_pixels = np.any(binary, axis=0)

    # Find digit boundaries (gaps between digits)
    in_digit = False
    digit_bounds = []
    start = 0

    for i, has_pixel in enumerate(cols_with_pixels):
        if has_pixel and not in_digit:
            start = i
            in_digit = True
        elif not has_pixel and in_digit:
            digit_bounds.append((start, i))
            in_digit = False

    if in_digit:
        digit_bounds.append((start, len(cols_with_pixels)))

    # Predict each digit
    predictions = []
    predicted_number = ""

    for left, right in digit_bounds:
        left = max(0, left - 2)
        right = min(img_array.shape[1], right + 2)
        digit_img = img_array[:, left:right]
        rows_with_pixels = np.any(digit_img > threshold, axis=1)
        if rows_with_pixels.any():
            top = np.where(rows_with_pixels)[0][0]
            bottom = np.where(rows_with_pixels)[0][-1]
            digit_img = digit_img[max(0,top-2):min(digit_img.shape[0],bottom+3), :]
        h, w = digit_img.shape
        if h > w:
            pad = (h - w) // 2
            digit_img = np.pad(digit_img, ((0, 0), (pad, h - w - pad)), mode='constant')
        elif w > h:
            pad = (w - h) // 2
            digit_img = np.pad(digit_img, ((pad, w - h - pad), (0, 0)), mode='constant')
        pil_img = Image.fromarray(digit_img.astype(np.uint8))
        pil_img = pil_img.resize((16, 16), Image.Resampling.LANCZOS)
        digit_array = np.array(pil_img, dtype=np.float64)
        digit_array = (digit_array / 255.0) * 2 - 1
        features = extract_enhanced_features(digit_array.flatten().reshape(1, -1), norm_stats=norm_stats)
        scores = classifier.predict_scores(features)[0]
        pred = np.argmax(scores)
        predictions.append((pred, scores))
        predicted_number += str(pred)
    return predicted_number, predictions

def test_on_dataset(classifier, norm_stats=None, dataset_file="ZipDigits.test", num_samples=10):
    """Test the classifier on random samples from the dataset."""
    data = np.loadtxt(dataset_file)
    labels = data[:, 0].astype(int)
    images = data[:, 1:]
    indices = np.random.choice(len(data), num_samples, replace=False)
    print(f"\nTesting {num_samples} random samples:")
    print("-" * 50)
    correct = 0
    for idx in indices:
        true_label = labels[idx]
        pred, scores = predict_digit(classifier, images[idx], norm_stats=norm_stats)
        status = "OK" if pred == true_label else "WRONG"
        if pred == true_label:
            correct += 1
        print(f"Sample {idx}: True={true_label}, Predicted={pred} [{status}]")
    print("-" * 50)
    print(f"Accuracy: {correct}/{num_samples} ({100*correct/num_samples:.1f}%)")

if __name__ == "__main__":
    import sys
    try:
        classifier, norm_stats = load_model("my_model.pkl")
    except FileNotFoundError:
        print("No saved model found. Please train first:")
        print("  1. Run TrainEnhanced.py")
        sys.exit(1)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nProcessing: {image_path}")
        try:
            number, preds = predict_multi_digit(classifier, image_path, norm_stats=norm_stats)
            print(f"\nPredicted number: {number}")
            for i, (digit, scores) in enumerate(preds):
                print(f"  Digit {i+1}: {digit} (confidence: {scores[digit]:.2f})")
        except Exception as e:
            pred, scores = predict_digit(classifier, image_path, norm_stats=norm_stats, show_preview=True)
            print(f"\nPredicted digit: {pred}")
            print(f"Confidence scores: {scores.round(2)}")
    else:
        print("\nNo image provided. Testing on dataset samples...")
        test_on_dataset(classifier, norm_stats=norm_stats, num_samples=10)
        print("\nUsage: python PredictDigit.py <image_path>")
