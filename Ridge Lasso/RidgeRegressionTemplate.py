# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    data = np.loadtxt(filename)
    x = data[:,:-1]
    y = data[:, -1]
    return x, y


# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    assert len(x) == len(y) #catch
    m = int(len(x) * train_proportion) #get proportion of total length to split on
    val = 100 - m
    x_train, x_test = x[:m], x[m:]#split X
    y_train, y_test = y[:m], y[m:]#split Y
    assert x_train.shape[0] == y_train.shape[0], "Lengths don't align on train set"
    assert x_test.shape[0] == y_test.shape[0], "Lengths don't align on test set"
    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python
def normal_equation(x, y, lambdaV):
    #basically the same as OLS, but just adding an augmented identity matrix to make it non singular
    beta = np.dot(np.linalg.inv(np.dot(x.T,x) + lambdaV * np.identity(x.shape[1])),np.dot(x.T, y))
    return np.asarray(beta)

# Extra Credit: Find theta using gradient descent
def gradient_descent(x, y, lambdaV, num_iterations, learning_rate):
    # your code
    return beta

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    loss = np.sum((y- y_predict)**2) * (1 / len(y)) #Mean Square Error
    return loss

# Given an array of x and theta predict y
def predict(x, theta):
    y_predict = x.dot(theta)
    return y_predict

# Find the best lambda given x_train and y_train using 4 fold cv
def cross_validation(x_train, y_train, lambdas):
    valid_losses = []
    training_losses = []
    # your code
    return np.array(valid_losses), np.array(training_losses)

if __name__ == "__main__":

    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt") # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss")
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss")
    plt.savefig('lamba_vs_loss')
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]


    # step 2: analysis
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = # your code get l2 norm of normal_beta
    best_beta_norm = # your code get l2 norm of best_beta
    large_lambda_norm = # your code get l2 norm of large_lambda_beta
    print(best_lambda)
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " + str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " + str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " + str(get_loss(y_test, predict(x_test, large_lambda_beta))))
    bar_plot(best_beta)


    # Step3: Retrain a new model using all sampling in training, then report error on testing set
    # your code !


    # Step Extra Credit: Implement gradient descent, analyze and show it gives the same or very similar beta to normal_equation
    # to prove that it works
