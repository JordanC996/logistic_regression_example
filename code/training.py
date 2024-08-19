from dataset import create_dataset, sigmoid
import numpy as np

def model_init(num_parameters: int):
    return np.random.normal(0, 0.1, num_parameters)

def training(num_epoch: int, learning_rate: float, num_parameters:int, mini_batch_size: int, data_points: np.ndarray, labels: np.ndarray):
    theta = model_init(num_parameters)
    num_mini_batch = len(data_points) // mini_batch_size

    for epoch in num_epoch:
        #shuffle
        random_indeces = np.random.choice(len(data_points), len(data_points), replace=False)
        shuffled_data_points = data_points[random_indeces]
        shuffled_labels = labels[random_indeces]
        x_batches = np.array_split(shuffled_data_points, num_mini_batch)
        y_batches = np.array_split(shuffled_labels, num_mini_batch)
        for x_batch,  y_batch in zip(x_batches, y_batches):
            update_step_value = np.zeros(num_parameters)
            for x, y in zip(x_batch, y_batch):
                model_inference = np.dot(theta, x)
                non_linear_model_inference = sigmoid(model_inference)
                update_step_value += (y - non_linear_model_inference)*x
            


