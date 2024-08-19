import numpy as np

def create_dataset(n_samples: int, n_features: int, mu = 0, sigma = 1):
    weights = np.random.uniform(-10, 10, n_features)
    bias = np.random.uniform(1, 10)
    x = np.random.normal(mu, sigma, (n_samples, n_features)) 
    linear_combination = np.dot(x, weights) + bias
    probabilities = sigmoid(linear_combination)
    y = (probabilities > 0.5).astype(int)
    noise_percentage = np.random.uniform(0,1)
    noise_labels_number = int(np.round(len(y)*noise_percentage))
    noise_indeces = np.random.choice(len(y), noise_labels_number, replace=False)
    y[noise_indeces] = 1 - y[noise_indeces]
    return x, y
    


def sigmoid(z):
    return 1/(1 + np.exp(-z))

