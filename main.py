import multiprocessing
import sys

import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from tensorflow import keras
import fed


EPOCHS = 10
ITERATIONS = 10
BATCH_SIZE = 64
NUM_PARTIES = 2
PARTIES = {'alice', 'bob'}

# Hyperparameters and model dimensions.
INPUT_DIM = 9
NUM_CLASSES = 1

@fed.remote
def create_model():
    """Defines a simple Keras model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(NUM_CLASSES, activation='linear')
    ])
    model.compile(optimizer='RMSprop',
                  loss='mean_squared_error',
                  metrics=['mse'])
    return model

@fed.remote
class Node:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(INPUT_DIM,)),
            tf.keras.layers.Dense(NUM_CLASSES, activation='linear')
            ])
        self.model.compile(optimizer='RMSprop',
                  loss='mean_squared_error',
                  metrics=['mse'])

    def load_data(self, data):
        """Imports dataset in node and does preprocessing."""

        data['FinancialStatus'] = data['FinancialEducation'] * np.log(data['Wealth'])

        # Define the target column (the column you want to predict)
        target_column = 'RiskPropensity'

        # Separate the features (X) and the target (y)
        X = data.drop(columns=[target_column])  # All columns except the target
        y = data[target_column]  # The target column

        # BoxCox transformation on Wealth and Income
        X['Wealth'], fitted_lambda_wealth = boxcox(X['Wealth'])
        X['Income'], fitted_lambda_income = boxcox(X['Income'])

        # Initialize a scaler
        scaler = StandardScaler()

        # Fit the scaler on X and transform X
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets (75% training, 25% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

    def train(self):
        print("training local model")
        self.model.fit(self.X_train, self.y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    def get_weights(self):
        print("getting weights")
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def evaluate(self):
        return  self.X_test, self.y_test

@fed.remote
def get_datasets(file_path):
    # Get dataset from CSV
    with open(file_path, 'r') as f:
        data = pd.read_csv(f)

    # Drop unnecessary tuple ID column
    data = data.drop('ID', axis=1)

    # Split dataset in NUM_PARTIES parts
    data_parties = np.array_split(data, NUM_PARTIES)

    return data_parties

@fed.remote
def avg_weights(all_weights):
    # Transpose the list of lists so that we group by layer:
    # e.g., weights_for_layer0 = [party1_layer0, party2_layer0, ...]
    averaged = []
    weights = fed.get(all_weights)
    for weights_per_layer in zip(*weights):
        # Each element in weights_per_layer is an array (e.g. the kernel for layer0, from each party)
        averaged.append(np.mean(weights_per_layer, axis=0))
    return averaged


def run(party):
    print(f"I am {party}")
    ray.init(address='local', include_dashboard=False)

    addresses = {
        'server': '127.0.0.1:11013',
        'alice': '127.0.0.1:11012',
        'bob': '127.0.0.1:11011',
    }
    # addresses = {
    #     '0': '127.0.0.1:11010',
    #     '1': '127.0.0.1:11011',
    #     '2': '127.0.0.1:11012',
    #     '3': '127.0.0.1:11013',
    #     '4': '127.0.0.1:11014',
    #     '5': '127.0.0.1:11015',
    #     '6': '127.0.0.1:11016',
    #     '7': '127.0.0.1:11017',
    #     '8': '127.0.0.1:11018',
    #     '9': '127.0.0.1:11019'
    # #    'server': '127.0.0.1:11020',
    # }
    fed.init(addresses=addresses,  party=party)

    # Parties definition
    parties = [Node.party(f"{i}").remote() for i in PARTIES]
    server = Node.party("server").remote()

    # Get dataset
    file_path = "/Users/antonelloamore/PycharmProjects/ray-fed-training/Needs.csv"
    datasets = get_datasets.party("server").remote(file_path)
    datasets = fed.get(datasets)

    # Load local data into node + data prep
    for i in range(NUM_PARTIES):
        parties[i].load_data.remote(datasets[i])

    global_weights = server.get_weights.remote()

    # FedAvg
    for i in range(ITERATIONS):
        print(f"Iteration {i+1}")

        # Local train
        for i in range(NUM_PARTIES):
            parties[i].set_weights.remote(global_weights)
            parties[i].train.remote()

        local_weights = [parties[i].get_weights.remote() for i in range(NUM_PARTIES)]
        print("pizza")
        w_mean = avg_weights.party("server").remote(local_weights)
        print("hamburger")

        global_weights = fed.get(w_mean)

    # Only server should do this
    server.set_weights.remote(global_weights)

    ### --- test
    X_global_test, y_global_test = fed.get(parties[0].evaluate.remote())
    global_loss = server.evaluate.remote(X_global_test, y_global_test, verbose=1)
    print(f"Global Model Evaluation - Loss: {fed.get(global_loss)}")
    ### --- end test

    fed.shutdown()
    ray.shutdown()

if __name__ == "__main__":
    p_alice = multiprocessing.Process(target=run, args=('alice',))
    p_bob = multiprocessing.Process(target=run, args=('bob',))
    p_server = multiprocessing.Process(target=run, args=('server',))
    p_alice.start()
    p_bob.start()
    p_server.start()
    p_alice.join()
    p_bob.join()
    p_server.join()