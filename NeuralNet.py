###############################
#   Neural Network Analysis   #
###############################


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPClassifier


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)

    def preprocess(self):
        # Drop rows with null values
        self.raw_input.dropna(inplace=True)

        # Standardize numeric features
        numeric_cols = self.raw_input.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        self.processed_data = pd.DataFrame(scaler.fit_transform(self.raw_input[numeric_cols]), columns=numeric_cols)

        # Convert categorical variables to numerical values using one-hot encoding
        categorical_cols = self.raw_input.select_dtypes(include='object').columns
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_cols = encoder.fit_transform(self.raw_input[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
        self.processed_data = pd.concat([self.processed_data, encoded_df], axis=1)

        return 0

    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot is color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols - 1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rates = [0.01, 0.1]
        max_iterations = [100, 200]  # also known as epochs
        num_hidden_layers = [2, 3]

        # Create all possible combinations of hyperparameters
        hyperparameters = list(itertools.product(activations, learning_rates, max_iterations, num_hidden_layers))

        # Initialize lists to store the results
        models = []
        train_accuracies = []
        train_errors = []
        test_accuracies = []
        test_errors = []

        # Create and evaluate each model
        for hyperparams in hyperparameters:
            # Create the neural network
            model = MLPClassifier(hidden_layer_sizes=(hyperparams[3],),
                                  activation=hyperparams[0],
                                  learning_rate_init=hyperparams[1],
                                  max_iter=hyperparams[2])

            # Train the model and store the history
            history = model.fit(X_train, y_train)
            models.append(model)

            # Evaluate the model on the training set
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_accuracies.append(train_accuracy)
            train_error = mean_squared_error(y_train, y_train_pred)
            train_errors.append(train_error)

            # Evaluate the model on the test set
            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_accuracies.append(test_accuracy)
            test_error = mean_squared_error(y_test, y_test_pred)
            test_errors.append(test_error)

            # Plot the history of the model
            plt.plot(history.loss_curve_, label=str(hyperparams))

        # Configure the plot
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model history')
        plt.legend(loc='upper right')
        plt.show()

        # Output the results in a table
        table_headers = ["Model(Activation, Learning rate, Epochs, Hidden layers)", "Train accuracy", "Train error",
                         "Test accuracy", "Test error"]
        table_data = []

        for i in range(len(hyperparameters)):
            row = [hyperparameters[i], train_accuracies[i], train_errors[i], test_accuracies[i], test_errors[i]]
            table_data.append(row)

        # Print the results in a formatted table
        print(tabulate(table_data, headers=table_headers, tablefmt="pretty"))

        return 0


if __name__ == "__main__":
    neural_network = NeuralNet(
        "https://raw.githubusercontent.com/SamH135/breast-cancer-data-file/main/wdbc%5B1%5D.data")
    neural_network.preprocess()
    neural_network.train_evaluate()
