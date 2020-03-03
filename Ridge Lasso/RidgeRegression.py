# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    data = np.loadtxt(filename)
    x = data[:,:-1]
    y = data[:, -1]
    return x, y

# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
#This is tested and works
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
    x,y = shuffle(x,y)
    train_losses = []
    val_losses = []
    x_train, x_val, y_train, y_val = train_test_split(x, y, 0.8)

    beta = np.random.random(x_train.shape[1])
    for i in range(num_iterations):
        beta =beta - (learning_rate / len(x_train)) * \
                (np.dot(x_train.T, (x_train.dot(beta) - y_train)) \
                 + 2 * lambdaV * beta )
        train_losses.append(get_loss(y_train, predict(x_train, beta)))
        val_losses.append(get_loss(y_val, predict(x_val, beta)))

    plt.plot([i for i in range(num_iterations)],train_losses)
    plt.plot([i for i in range(num_iterations)],val_losses)
    plt.xscale("log")
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.title('Training/Validation Loss, Ridge Gradient Descent')
    plt.legend(['train loss', 'val loss'])
    plt.plot()
    return beta, train_losses, val_losses

def get_loss(y, y_predict):
    loss = np.sum((y- y_predict)**2) * (1 / len(y)) #Mean Square Error
    return loss

def predict(x, theta):
    y_predict = x.dot(theta)
    return y_predict

# Find the best lambda given x_train and y_train using 4 fold cv
def cross_validation(x_train, y_train,lambdaV):
    valid_losses = []
    training_losses = []
    k = 4
    #try catching for remainder/uneven split
    m = len(x)
    extras = m % 4 #get remainder of length of set divided by number of folds
    if extras != 0:
        x_train_extras = x_train[-extras:] #store for later
        y_train_extras = y_train[-extras:] #store for later
        x_train = x_train[:-extras] #discard extras
        y_train = y_train[:-extras]#discard extras

    #split the data into 4 equal sized array
    x_train_split = np.split(x_train, k)
    y_train_split = np.split(y_train, k)
    sets = []
    for i in range(k):
        #loop through k folds
        #take a split of the data for testing
        X_test, y_test = x_train_split[i], y_train_split[i]
        #take the remaining 3/4 of the data for training
        X_train = np.concatenate(x_train_split[:i] + x_train_split[i + 1:], axis=0)
        y_train = np.concatenate(y_train_split[:i] + y_train_split[i + 1:], axis=0)
        sets.append([X_train, X_test, y_train, y_test])

    #try catching for remainder/uneven split. adding to last set as training example
    if extras != 0:
        np.append(sets[-1][0], x_train_extras, axis=0)
        np.append(sets[-1][2], y_train_extras, axis=0)

    for i in lambdaV:
        temp_train_loss = []
        temp_test_loss = []
        for j in sets:
            thetas = normal_equation(j[0], j[2], i)
            temp_train_loss.append(get_loss(j[2], predict(j[0], thetas)))
            temp_test_loss.append(get_loss(j[3], predict(j[1], thetas)))
        training_losses.append(np.mean(temp_train_loss)) #get mean over all folds
        valid_losses.append(np.mean(temp_test_loss)) #get mean over all folds
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
    plt.title("lambda vs training and validation loss")
    plt.show()
    #plt.savefig('lamba_vs_loss')

    best_lambda = lambdas[np.argmin(valid_losses)]
    # step 2: analysis
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = np.round(np.sqrt(np.dot(normal_beta, normal_beta)),2)
    best_beta_norm = np.round(np.sqrt(np.dot(best_beta, best_beta)),2)
    large_lambda_norm = np.round(np.sqrt(np.dot(large_lambda_beta, large_lambda_beta)),2)
    print(best_lambda)
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta: {:10.4f}".format(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta: {:10.4f}".format(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta: {:10.4f}".format(get_loss(y_test, predict(x_test, large_lambda_beta))))
    plt.bar([i for i in range(102)], best_beta)
    plt.title("Bar Plot of Feature Importance by X index")
    plt.show()



    # Step3: Retrain a new model using all sampling in training, then report error on testing set
    # your code !
    best_beta = normal_equation(x_train, y_train, best_lambda)
    train_predictions = predict(x_train, best_beta)
    test_predictions = predict(x_test, best_beta)
    train_loss = get_loss(y_train, train_predictions)
    test_loss = get_loss(y_test, test_predictions)
    print("Training Loss w/ best beta: {:10.4f}".format(train_loss))
    print("Test Loss w/ best beta : {:10.4f}".format(test_loss))

    # Step Extra Credit: Implement gradient descent, analyze and show it gives the same or very similar beta to normal_equation
    # to prove that it works
    beta, train_losses, val_losses = gradient_descent(x_train, y_train, lambdaV = 4,
                     num_iterations = 20000
                               , learning_rate =0.001)

    gd_loss = get_loss(y_test, predict(x_test, beta))
    print('gradient descent, betas:',np.round(beta[:5], 3))
    print("gradient descent test loss :", np.round(gd_loss,3))
    print('normal equation, betas:',np.round(best_beta[:5], 3))
    plt.close()
    X = [i for i in range(5)]
    Y =  beta[:5]
    Z = best_beta[:5]
    _X = np.arange(len(X))
    plt.bar(_X - 0.2, Y, 0.4)
    plt.bar(_X + 0.2, Z, 0.4)
    plt.xticks(_X, X) # set labels manually
    plt.legend(['gradient descent', 'normal equation'])
    plt.xlabel('feature index')
    plt.ylabel('Beta Value')
    plt.show()
