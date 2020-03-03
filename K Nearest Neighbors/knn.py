# Starting code for UVA CS 4501 Machine Learning- KNN

import numpy as np
np.random.seed(37)
# for plot
import matplotlib.pyplot as plt
#more imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

## the only purpose of the above import is in case that you want to compare your knn with sklearn knn

from collections import Counter
import operator

# Load file into np arrays
# x is the features
# y is the labels
def read_file(file):
    data = np.loadtxt(file, skiprows=1)
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y

# 2. Generate the i-th fold of k fold validation
# Input:
# x is an np array for training data
# y is an np array for labels
# i is an int indicating current fold
# nfolds is the total number of cross validation folds

def fold(x, y ,i, nfolds):
    valid_losses = []
    training_losses = []
    k = nfolds

    #try catching for remainder/uneven split
    m = len(x)
    extras = m % nfolds #get remainder of length of set divided by number of folds

    if extras != 0:
        x_train_extras = x[-extras:] #store for later
        y_train_extras = x[-extras:] #store for later

        x_train = y[:-extras] #discard extras
        y_train = y[:-extras]#discard extras

    #split the data into k equal sized array
    x_train_split = np.split(x, k)
    y_train_split = np.split(y, k)

    sets = []

    for i in range(k):
        #loop through k folds
        #take a split of the data for testing
        x_test, y_test = x_train_split[i], y_train_split[i]
        #take the remaining 3/4 of the data for training
        x_train = np.concatenate(x_train_split[:i] + x_train_split[i + 1:], axis=0)
        y_train = np.concatenate(y_train_split[:i] + y_train_split[i + 1:], axis=0)
        sets.append([x_train, x_test, y_train, y_test])

    #try catching for remainder/uneven split. adding to last set as training example
    if extras != 0:
        np.append(sets[-1][0], x_train_extras, axis=0)
        np.append(sets[-1][2], y_train_extras, axis=0)

    return sets[i] #return only a single specified fold. Will need a nested loop within findbestK to iterate through all folds to compute MSE on test sets
#Distance metric to rank nearest neighbors
def euclidean_distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

# 3. Classify each testing points based on the training points
# Input
# x_train: a numpy array of training data
# x_test: a numpy array
# k: the number of neighbors to take into account when predicting the label
# Output
# y_predict: a numpy array
def classify(x_train, y_train, x_test, k):
    neighbors = {} #storafe
    for i in range(len(x_test)): #loop through each test instance
        distances = []
        for j in range(len(x_train)):
            distances.append((j, euclidean_distance(x_test[i], x_train[j]), y_train[j])) #for each training instance, compute distance
        sorted_all = sorted(distances, key = lambda kv: kv[1]) # sort all by distance
        neighbors[i] = sorted_all[:k] #truncate to only include k neighbors

    y_predict = [] #perform inference
    for i in range(len(x_test)):
        neighbors_y = Counter([x[2] for x in neighbors[i]]) #get responses for each test instances nearest neighbors
        y_predict.append(max(neighbors_y.items(), key=operator.itemgetter(1))[0]) #find max of responses (maj vote)
    # Euclidean distance as the measurement of distance in KNN
    return np.array(y_predict)

# 4. Calculate accuracy by comaring with true labels
# Input
# y_predict is a numpy array of 1s and 0s for the class prediction
# y is a numpy array of 1s and 0s for the true class label
def calc_accuracy(y_predict, y_true):
    acc = np.sum(y_true == y_predict, axis=0) / len(y_true) #count where reponses match divided by len of y_test
    return acc

# 5. Draw the bar plot of k vs. accuracy
# klist: a list of values of ks
# accuracy_list: a list of accuracies
def barplot(klist, accuracy_list):
    plt.bar(klist, accuracy_list)
    plt.title('# of Neighbors Vs CV Accuracy')
    plt.xlabel('# of neighbors')
    plt.ylabel('Accuracy')
    plt.show()


# 1. Find the best K
def findBestK(x, y, klist, nfolds):
    kbest = 0
    best_acc = 0
    counter = 0
    accuracy_list = []
    for k in klist: #itereate through k neighbors
        fold_acc = []
        for i in range(0, nfolds - 1): #iterate through each fold
            x_train, x_test, y_train, y_test = fold(x,y, i, nfolds) #different for each fold
            y_predict = classify(x_train, y_train, x_test, k=k) #inference on test set for each fold
            accuracy = calc_accuracy(y_predict, y_test) #accuracy on test set for each fold
            fold_acc.append(accuracy) #store

        accuracy_list.append(np.mean(accuracy)) #take the average accuracy across all folds for each iteration of k
        accuracy = accuracy_list[counter] #accuracy at current timestep

        if accuracy > best_acc: #will update after 1st run, then subject to logic
            kbest = k #store best neighbor param
            best_acc = accuracy
        print('Neighbors: {}, cv accuracy {}'.format(k, accuracy))

        counter += 1
    # plot cross validation error for each k : implement function barplot(klist, accuracy_list)
    barplot(klist, accuracy_list)
    return kbest


if __name__ == "__main__":
    filename = "Movie_Review_Data.txt"
    # read data
    x, y = read_file(filename)
    print('Num Features', x.shape[1])
    print('Num Obs:', x.shape[0])
    print('Count of Postive Class labels:', Counter(y)[1])
    print('Count of Negative Class labels:', Counter(y)[0])
    nfolds = 8
    klist = [3, 5,7, 9, 11, 13]
    # Implementation covers two tasks, both part of findBestK function
    # Task 1 : implement kNN classifier for a given x,y,k
    # Task 2 : implement 4 fold cross validation to select best k from klist

    findBestK(x, y, klist, nfolds)
    print('-' *25)
    print('Testing with Sklearn, neighbors = 11')
    model = KNeighborsClassifier( 11)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = calc_accuracy(y_pred, y_test)
    print('sklearn KNN accuracy:', acc)
    # report best k, and accuracy, discuss why some k work better than others
