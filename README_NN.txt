HOW TO RUN:



The Easy Way: 

Copy paste the code in the NeuralNet.py file into a google colab code box and click "Run"



The Less Easy Way:

If you're not running in google colab you may need to install the following libraries as well as a working Python 3 IDE...

pip install numpy
pip install pandas
pip install tabulate
pip install matplotlib
pip install scikit-learn
pip install --upgrade scikit-learn

Once all the proper libraries are installed, simply press "Run" in your IDE and let the model train 
and display it's output.





NOTICE: 

There will likely be non-fatal error messages indicating that some of the models haven't converged yet but it will still
produce output (a graph of the different models and a table of the required information).

Also, if you run in something besides google colab, it may not produce the table until after you click X on the graph image rendering (which allows the rest of the program to print to console).





MODIFICATIONS:


The preprocess() method does the following:

1. Removes rows with missing data from the self.raw_input DataFrame.
2. Standardizes numeric columns, ensuring they have a mean of 0 and standard deviation of 1.
3. Converts categorical columns into numerical values using one-hot encoding.
4. Concatenates the standardized numeric and one-hot encoded categorical data into self.processed_data.
5. Returns 0 (a placeholder value).



The train_evaluate() method performs the following:

Splits the processed data into training and testing sets.
Specifies hyperparameters for a neural network model (e.g., activation functions, learning rates, epochs, and hidden layers).
Generates all possible combinations of these hyperparameters.
Initializes lists to store model results.
Iterates through the hyperparameter combinations and does the following for each combination:
Creates a neural network model with the specified hyperparameters.
Trains the model on the training data and records its training history.
Evaluates the model's performance on the training and testing data, calculating accuracy and error metrics.
Plots the training loss history for the model.
Prints a table of results, displaying hyperparameters and corresponding performance metrics.
Returns 0 as a placeholder value.