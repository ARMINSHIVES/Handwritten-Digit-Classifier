import numpy as np
from NeuralNetwork import NeuralNetwork  # your NN class

def compute_intensity(image):
    return np.mean(image)

def compute_symmetry(image):
    im = image.reshape(16, 16)
    left, right = im[:, :8], np.fliplr(im[:, 8:])
    return -np.mean((left - right) ** 2)

def normalize_feature(x):
    x_min, x_max = np.min(x), np.max(x)
    return (x - (x_max+x_min)/2)/((x_max-x_min)/2)

def polynomial_kernel(x, y):
    x1, x2 = x
    y1, y2 = y
    a = x1 * y1 + x2 * y2
    b = 1 + a
    return b**8


print("Loading ZIP digit data ...")
train = np.loadtxt("ZipDigits.train")
test = np.loadtxt("ZipDigits.test")
data = np.vstack((train, test))
labels = data[:,0]
images = data[:,1:]
intensity = np.array([compute_intensity(i) for i in images])
symmetry  = np.array([compute_symmetry(i) for i in images])
X = np.column_stack((normalize_feature(intensity),normalize_feature(symmetry)))
N = len(X)
idx = np.random.choice(N, 500, replace=False)
mask = np.ones(N, dtype=bool)
mask[idx] = False
X_D, y_D = X[idx], labels[idx]
X_test, y_test = X[mask], labels[mask]
y_D = np.where(y_D==1,1,-1)
y_test = np.where(y_test==1,1,-1)

NN_varlr = NeuralNetwork(m=10, output_activation="identity", rand_init=True)
Ein_history_varlr = NN_varlr.train(X_D, y_D, epochs=5000000)
NN_varlr.output_activation = "sign"
NN_varlr.plot_boundary(X_D, y_D, 500, "decision_boundary_partA.png")