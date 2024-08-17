import numpy as np

def create_dataset(n_samples: int, n_features: int, mu = 0, sigma = 1):
    weights = np.random.uniform(-10, 10, n_features)
    bias = np.random.uniform(1, 10)
    x = np.random.normal(mu, sigma, (n_samples, n_features)) 
    linear_combination = np.dot(x, weights) + bias
    probabilities = sigmoid(linear_combination)
    y = (probabilities > 0.5).astype(int)
    return x, y
    


def sigmoid(z):
    return 1/(1 + np.exp(-z))


