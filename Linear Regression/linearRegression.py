# Machine Learning HW1
#Jacob Dineen 
#JD5ED

import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams


############################################################################################################
# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    data = np.loadtxt(filename, delimiter="\t")
    #x contains two features - Column one is the bias vector
    #y contains actual output - Continuous
    x, y = data[:, :2], data[:, -1]
    return x, y


# Find theta using the normal equation
def normal_equation(x, y):
    #(aTa)^-1 (aTb)
    theta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    return theta


# Find thetas using stochiastic gradient descent
# Don't forget to shuffle
def stochiastic_gradient_descent(x,
                                 y,
                                 learning_rate,
                                 num_iterations,
                                 shuffle=True):
    thetas = []
    theta = np.random.random(x.shape[1])
    if shuffle:
        data = np.column_stack((x, y))
        np.random.shuffle(data)
        x, y = data[:, :2], data[:, -1]

    for i in range(num_iterations):
        for i in range(len(x)):
            random_datapoint = np.random.randint(0, len(x))
            x_i = x[random_datapoint]
            y_i = y[random_datapoint]
            grad = np.dot((x_i.dot(theta) - y_i), (x[i])) * (1.0 / len(x))
            theta = theta - (learning_rate * grad)
        thetas.append(theta)
    # your code
    return thetas


# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_iterations):
    thetas = []
    theta = np.random.random(x.shape[1])
    for i in range(num_iterations):
        #Batch training over full data set
        grad = (x.dot(theta) - y).dot(x) * (1.0 / len(x))
        theta = theta - (learning_rate * grad)
        thetas.append(theta)
    return thetas


# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x,
                               y,
                               learning_rate,
                               num_iterations,
                               batch_size,
                               shuffle=True):
    thetas = []
    theta = np.random.random(x.shape[1])
    num_batches = int(len(x) / batch_size)

    for i in range(num_iterations):
        if shuffle:
            data = np.column_stack((x, y))
            np.random.shuffle(data)
            x, y = data[:, :2], data[:, -1]
        #Randomly Shuffle before each epoch
        random_indices = np.random.permutation(len(x))
        x = x[random_indices]
        y = y[random_indices]
        for i in range(0, len(x), batch_size):
            x_i = x[i:i + batch_size]
            y_i = y[i:i + batch_size]
            grad = np.dot(x_i.dot(theta) - y_i, (x_i)) * (1.0 / len(x))
            theta = theta - (learning_rate * grad)
        thetas.append(theta)
    # your code
    return thetas


# Given an array of x and theta predict y
def predict(x, theta):
    y_predict = x.dot(theta)
    return y_predict


# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    loss = np.sum((y_predict - y)**2) * (1.0 / len(y))
    return loss


# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas, title, save=False):
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_loss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.scatter(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    if save:
        plt.savefig("{}".format(title.replace(" ", "")))
    plt.show()


def batch_size_graph_loss(x, y, batches, save=False):
    empty_dict = {}
    epochs = []
    losses = []
    for i in batches:
        epochs = []
        losses = []
        thetas = minibatch_gradient_descent(x, y, 0.1, 100, i)
        epoch_num = 1
        for theta in thetas:
            y_predict = predict(x, theta)
            losses.append(get_loss(y, y_predict))
            epochs.append(epoch_num)
            epoch_num += 1
        empty_dict[i] = (epochs, losses)
    for i in empty_dict.keys():
        plt.plot(empty_dict[i][0], empty_dict[i][1])
    plt.legend(batches)
    plt.title("Convergence Rate by Batch Size")
    if save:
        plt.savefig("Convergence_Batch")

    plt.show()


# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title, save=False):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    if save:
        plt.savefig("{}".format(title.replace(" ", "")))
    plt.show()


def plot_learningrate(model, model_str, x, y, save=False):
    thetas = []
    for i in learning_rates:
        try:
            theta = model(x, y, i, 100, 64)
            thetas.append(theta[-1])
        except:
            theta = model(x, y, i, 100)
            thetas.append(theta[-1])

    predictions = []
    for i in thetas:
        predictions.append(predict(x, i))

    plt.scatter(x[:, 1], y)
    for i in predictions:
        plt.plot(x[:, 1], i)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend([0.001, 0.005, 0.01, 0.05, 0.1, 0.3], loc='lower right')
        plt.title(model_str)
    if save:
        plt.savefig(model_str.replace(" ", ""))
    plt.show()


def batch_size_graph(x, y, batches, save=False):
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    losses = []
    for i in batches:
        thetas = minibatch_gradient_descent(x, y, 0.1, 100, i)
        y_predict = predict(x, thetas[-1])
        losses.append(get_loss(y, y_predict))
        plt.plot(x[:, 1], y_predict)
        plt.legend(batches)
    plt.title("MiniBatch SGD - Varied Batchsizes")
    if save:
        plt.savefig("MiniBatchSGDVariedBatchsizes")
    plt.show()

    for i, j in zip(batches, losses):
        print("Batch Size: {} Cost: {}".format(i, np.round(j, 3)))


if __name__ == "__main__":
############################################################################################################
	try:
		rcParams['figure.figsize'] = 5, 5
		x, y = load_data_set('regression-data.txt')
		# plot
		plt.scatter(x[:, 1], y)
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title("Scatter Plot of Data")
		plt.savefig("Scatter Plot of Data")
		plt.show()

		theta = normal_equation(x, y)
		print("Outputting Learned Function:")
		print("f(X) = (%.4f) + (%.4f)x" % (theta[0], theta[1]))
		plot(x, y, theta, "Normal Equation Best Fit")

		thetas = gradient_descent(
			x, y, 0.1,
			100)  # Try different learning rates and number of iterations
		print("Outputting Learned Function:")
		print("f(X) = (%.4f) + (%.4f)x" % (thetas[-1][0], thetas[-1][1]))
		plot(x, y, thetas[-1], "Gradient Descent Best Fit")
		plot_training_errors(x, y, thetas,
							 "Gradient Descent Mean Epoch vs Training Accuracy")

		thetas = stochiastic_gradient_descent(
			x, y, 0.1,
			100)  # Try different learning rates and number of iterations
		print("Outputting Learned Function:")
		print("f(X) = (%.4f) + (%.4f)x" % (thetas[-1][0], thetas[-1][1]))

		plot(x, y, thetas[-1], "Stochiastic Gradient Descent Best Fit")
		plot_training_errors(
			x, y, thetas,
			"Stochiastic Gradient Descent Mean Epoch vs Training Accuracy")

		thetas = minibatch_gradient_descent(x, y, 0.1, 100, 64)
		print("Outputting Learned Function:")
		print("f(X) = (%.4f) + (%.4f)x" % (thetas[-1][0], thetas[-1][1]))

		plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
		plot_training_errors(
			x, y, thetas,
			"Minibatch Gradient Descent Mean Epoch vs Training Accuracy")

		learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3]
		models = [
			gradient_descent, stochiastic_gradient_descent,
			minibatch_gradient_descent
		]
		model_str = [
			"gradient_descent", "stochiastic_gradient_descent",
			"minibatch_gradient_descent"
		]

		batch_size_graph(x, y, batches=[2, 4, 8, 64, 128, 256, 512, 1024])
		rcParams['figure.figsize'] = 10, 10
		batch_size_graph_loss(x, y, batches=[2, 4, 8, 64, 128, 256, 512, 1024])
		plot_learningrate(models[0], model_str[0], x, y)
		plot_learningrate(models[1], model_str[1], x, y)
		plot_learningrate(models[2], model_str[2], x, y)
	except:
		print(
			"Invalid Hyperparameters. Learning Rate and Epochs can't be None. Please adjust and rerun"
		)

