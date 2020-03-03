# Machine Learning HW2 Poly Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

# Step 1
# Parse the file and return 2 numpy arrays
def load_data_set(filename, random_state = 0):
    data = np.loadtxt(filename)
    x = data[:,0]
    y = data[:, -1]
    return x, y

def scale(data):
        return (data - np.mean(data))/(np.max(data) - np.min(data))

# Step 2:
# result: polynomial basis based reformulation of x
def increase_poly_order(x, degree):
    arr = np.array([x** i for i in range(0, degree + 1)]).T
    return arr


#This is tested and works
def train_test_split(x, y, train_proportion):
    assert len(x) == len(y) #catch
    m = int(len(x) * train_proportion) #get proportion of total length to split on
    x_train, x_test = x[:m], x[m:]#split X
    y_train, y_test = y[:m], y[m:]#split Y
    assert x_train.shape[0] == y_train.shape[0], "Lengths don't align on train set"
    assert x_test.shape[0] == y_test.shape[0], "Lengths don't align on test set"
    return x_train, x_test, y_train, y_test

#
def normal_equation(x, y):
    #(aTa)^-1 (aTb)
    theta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    return theta

def solve_regression(x, y, learning_rate, num_iterations):
    x,y = shuffle(x,y)
    thetas = []
    theta = np.zeros(x.shape[1])
    for i in range(num_iterations):
        #Batch training over full data set
        grad = (x.dot(theta) - y).dot(x) * (1.0 / len(x))
        theta = theta - (learning_rate * grad)
        thetas.append(theta)
    return theta, thetas


# Given an array of y and y_predict return loss
#NOTE: Using MSE as a loss here - Error looks like it is exploding during training due to the the
#power term. In reality, we're not too far off from output predictions.
def get_loss(y, y_predict):
    loss = np.sum((y- y_predict)**2) * (1 / len(y)) #Mean Square Error
    return loss


def predict(x, theta):
    y_predict = x.dot(theta)
    return y_predict


# Given a list of thetas one per (s)GD epoch
# this creates a plot of epoch vs prediction loss (one about train, and another about test)
# this figure checks GD optimization traits of the best theta
def plot_epoch_losses(x_train, x_test, y_train, y_test, best_thetas, title, save_figs = True):
    epochs = []
    training_losses = []
    testing_losses = []
    epoch_num = 1
    for theta in best_thetas:
        training_losses.append(get_loss(y_train, predict(x_train, theta)))
        testing_losses.append(get_loss(y_test, predict(x_test, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs,training_losses)
    plt.plot(epochs,testing_losses)
    plt.title(title)
    plt.legend(['training_loss', 'testing_loss'])
    plt.savefig('images\\epoch_loss_plot_traintest')
    plt.show()


# Output:
# training_losses: a list of losses on the training dataset
# validation_losses: a list of losses on the validation dataset
def get_loss_per_poly_order(x, y, degrees):
    training_losses = []
    validation_losses = []
    for i in degrees:
        x_new = increase_poly_order(x, degree =i)
        x_train, x_val, y_train, y_val = train_test_split(x_new, y, train_proportion = 0.75)
        #train model
        theta = normal_equation(x_train, y_train)
        training_losses.append(get_loss(x_train.dot(theta), y_train))
        validation_losses.append(get_loss(x_val.dot(theta), y_val))
    # your code
    return training_losses, validation_losses

# Give the parameter theta, best-fit degree , plot the polynomial curve
def best_fit_plot(degree, save = True):
    x, y = load_data_set("dataPoly.txt")
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_proportion = 0.8)
    x_train = increase_poly_order(x_train, 8)
    normal_theta_8 = normal_equation(x_train, y_train)
    x_new = np.linspace(min(x), max(x), 200)
    x_new_1 = increase_poly_order(x_new, 8)
    y_new = x_new_1.dot(normal_theta_8)
    plt.plot(x,y,'o', x_new, y_new)
    plt.title('best_poly_fit')
    if save:
        plt.savefig('images\\best_poly_fit')
    plt.show()
    print('best theta via normal equation: {}'.format(np.round(normal_theta_8,3)))

    # your code


def select_hyperparameter(degrees, x_train, x_test, y_train, y_test, save_figs = True):
    # Part 1: hyperparameter tuning:
    # Given a set of training examples, split iinto train-validation splits
    # do hyperparameter tune
    # come up with best model, then report error for best model
    training_losses, validation_losses = get_loss_per_poly_order(x_train, y_train, degrees) #using normal equation
    #On the above step, we split out train sets into train and val sets. left test sets for later
    plt.plot(degrees, training_losses, label="training_loss")
    plt.plot(degrees, validation_losses, label="validation_loss")
    print("min train loss: {}".format(min(training_losses)))
    print("min validation loss: {}".format(min(validation_losses)))
    plt.yscale("log") #scale down axis
    plt.legend(loc='best')
    plt.title("poly order vs validation_loss")
    if save_figs:
        plt.savefig("images\\Training_Validation_Losses_Best_D")
    plt.show()

    # Part 2:  testing with the best learned theta
    # Once the best hyperparameter has been chosen
    # Train the model using that hyperparameter with all samples in the training
    # Then use the test data to estimate how well this model generalizes.
    best_degree = 8
    x_train = increase_poly_order(x_train, best_degree) #scale here to avoid overflow
    x_test = increase_poly_order(x_test, best_degree)
    best_theta, best_thetas = solve_regression(x_train, y_train, 0.0001, 2000) #using batch SGD
    best_fit_plot(degree = 8)
    print("Best Theta from Batch GD: {}".format(best_theta))
    test_loss = get_loss(y_test, predict(x_test, best_theta))
    train_loss = get_loss(y_train, predict(x_train, best_theta))
    # Part 3: visual analysis to check GD optimization traits of the best theta
    plot_epoch_losses(x_train, x_test, y_train, y_test, best_thetas, "best learned theta - train, test losses vs. GD epoch ")
    return best_degree, best_theta, train_loss, test_loss


# Given a list of dataset sizes [d_1, d_2, d_3 .. d_k]

def get_loss_per_tr_num_examples(x, y, example_num, train_proportion):
    training_losses = []
    testing_losses = []
    for i in example_num:
        x_train, x_test, y_train, y_test = train_test_split(x[:i], y[:i], train_proportion = train_proportion)
        theta = normal_equation(x_train, y_train)
        train_loss = get_loss(y_train, predict(x_train, theta))
        test_loss = get_loss(y_test, predict(x_test, theta))
        training_losses.append(train_loss)
        testing_losses.append(test_loss)
    return training_losses, testing_losses


if __name__ == "__main__":

    # select the best polynomial through train-validation-test formulation
    x, y = load_data_set("dataPoly.txt")
    #We split the full dataset twice to ensure that we have a 60/20/20 train/validation/test split
    #as per the instructions in the homework. This is done separately in the get_loss_per_poly_order method
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_proportion = 0.8)

    degrees = [i for i in range(10)]
    #We split the data again in the hyperparameter method to avoid tuning on test loss
    best_degree, best_theta, train_loss, test_loss = select_hyperparameter(degrees, x_train, x_test, y_train, y_test)

    # Part 4: analyze the effect of revising the size of train data:
    # Show training error and testing error by varying the number for training samples
    x, y = load_data_set("dataPoly.txt")
    x = increase_poly_order(x, 8)
    example_num = [10*i for i in range(2, 11)] # python list comprehension
    training_losses, testing_losses = get_loss_per_tr_num_examples(x, y, example_num, 0.5)
    plt.plot(example_num, training_losses, label="training_loss")
    plt.plot(example_num, testing_losses, label="testing_losses")
    plt.yscale("log")
    plt.legend(loc='best')
    plt.title("number of examples vs training_loss and testing_loss")
    #plt.savefig("images\\examples_vs_traintest_loss")
    plt.show()
