# Starting code for UVA CS 4501 ML- SVM

# Starting code for UVA CS 4501 ML- SVM

import numpy as np
np.random.seed(37)
import random
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Att: You're not allowed to use modules other than SVC in sklearn, i.e., model_selection.

# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']


# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing.
# For example, as a start you can use one hot encoding for the categorical variables and normalization
# for the continuous variables.
def calc_accuracy(y_predict, y_true):
    acc = np.sum(y_true == y_predict, axis=0) / len(y_true) #count where reponses match divided by len of y_test
    return acc



def load_data(csv_file_path, encode_cat = False, outlier_removal = True, feature_engineering = True, down_sample = False):
    print('reading in data')
    #I believe pandas is allowed. Using it for easier recognition of column names
    data = pd.read_csv(csv_file_path, names = col_names_x + col_names_y)
    df = data.copy()
    df['label']  = df['label'].apply(lambda x: 0 if x==' <=50K' else 1)
    #df['native-country']  = df['native-country'].apply(lambda x: "Other" if x!= " United-States" else "United-States")
    df.drop(['native-country'], axis=1, inplace = True)
    if feature_engineering:
        df["marital-status"] = df["marital-status"].replace([' Married-civ-spouse',' Married-spouse-absent',' Married-AF-spouse'], 'Married')
        df["marital-status"] = df["marital-status"].replace([' Never-married',' Divorced',' Separated',' Widowed'], 'Single')
        df['race'] = df['race'].map({' White': 1, ' Asian-Pac-Islander': 1, ' Black':0, ' Amer-Indian-Eskimo':0, ' Other':0})
        df['relationship'] = df['relationship'].map({' Not-in-family':0, ' Unmarried':0, ' Own-child':0, ' Other-relative':0, ' Husband':1, ' Wife':1})
        #df.drop(categorical_cols, axis=1, inplace = True)
    if outlier_removal:
        print('removing outliers')
        df = df[(np.abs(stats.zscore(df[numerical_cols])) < 3).all(axis=1)]


    if encode_cat: #encode cat features / get dummies
        print('Categorical one hot encoding')
        df = pd.get_dummies(df)

    x = df.drop(['label'], axis=1) #extract ind. features
    y = df['label']#extract response
    if down_sample:
        x = x[:10000]
        y = y[:10000]
    return x,y, x.columns

def cross_validation(x_train, y_train, xcols, params, num_folds=3):

    #Split Data into k sets. 'Sets' contains set of sets
    ######################################################################################
    k = num_folds
    #try catching for remainder/uneven split
    m = len(x_train)
    extras = m % num_folds  #get remainder of length of set divided by number of folds
    if extras != 0:
        x_train_extras = x_train[-extras:]
        y_train_extras = y_train[-extras:]  #store for later
        x_train = x_train[:-extras]  #discard extras
        y_train = y_train[:-extras]  #discard extras
    #print(len(x_train))
    #print(len(y_train))
    x_train_split = np.split(x_train, k)
    y_train_split = np.split(y_train, k)

    sets = []
    for i in range(k):
        #loop through k folds
        #take a split of the data for testing
        X_test = pd.DataFrame(x_train_split[i],
                              columns =xcols)
        y_test = y_train_split[i]
        #take the remaining 3/4 of the data for training
        X_train = pd.DataFrame(np.concatenate(x_train_split[:i] +
                                              x_train_split[i + 1:],
                                              axis=0),
                               columns =xcols)
        y_train = np.concatenate(y_train_split[:i] +
                                              y_train_split[i + 1:],
                                              axis=0)
        sets.append([X_train, X_test, y_train, y_test])
    print('number of sets:', len(sets))
    #try catching for remainder/uneven split. adding to last set as training example
    #if extras != 0:

    #    np.append(sets[-1][0], x_train_extras, axis=0)
    #    np.append(sets[-1][2], y_train_extras, axis=0)
    ######################################################################################
    #Grid search + KFOLD
    ######################################################################################
    train_acc = []
    valid_acc = []
    models = []
    counter = 0
    for i in params:
        temp_test_acc = []
        temp_train_acc = []
        model = SVC(C=i['C'],
                    kernel=i['kernel'],
                    degree=i['degree'],
                    gamma='auto')
        models.append(model)
        fold = 1
        print(i)
        for j in sets:
            scaler = MinMaxScaler().fit((j[0])) #fit minmax scaler to trainset
            clean_x_train = pd.DataFrame(scaler.transform(j[0]), columns = xcols)  #scale

            model.fit(clean_x_train, j[2])  #fit on cleaned/scaled train set

            clean_x_test = pd.DataFrame(scaler.transform(j[1]), columns = xcols)  #clean/scale test set

            train_preds = model.predict(clean_x_train)
            test_preds = model.predict(clean_x_test)  #predict on clean/scaled test set

            train_accuracy = calc_accuracy(j[-2], train_preds)
            test_accuracy = calc_accuracy(j[-1], test_preds)

            temp_train_acc.append(train_accuracy)
            temp_test_acc.append(test_accuracy)

            fold += 1

        train_acc.append(np.mean(temp_train_acc))  #get mean over all folds
        valid_acc.append(np.mean(temp_test_acc))  #get mean over all folds
        print('mean accuracy on train set across folds:', train_acc[counter])
        print('mean accuracy on holdout set across folds:', valid_acc[counter])
        fold = 1
        counter += 1
    df = pd.DataFrame(params)
    df['cv_train_accuracy'] = train_acc
    df['cv_test_accuracy'] = valid_acc
    best_params = params[df['cv_test_accuracy'].idxmax()]
    return df, df['cv_test_accuracy'].max(), models[df['cv_test_accuracy'].idxmax()], best_params



# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(training_csv, submission):
    # load data and preprocess from filename training_csv
    if submission:
        x,y, xcols = load_data(training_csv, encode_cat = True, outlier_removal = True,
                               feature_engineering = True, down_sample= True)
        print('Turned in Version is downsampled to include only 10k observations')

    else:
        x,y, xcols = load_data(training_csv, encode_cat = True, outlier_removal = True,
                               feature_engineering = True, down_sample= False)
    # hard code hyperparameter configurations, an example:
    param_set = [
                    # {'kernel': 'rbf', 'C': .1, 'degree': 1},
                    #{'kernel': 'rbf', 'C': 1, 'degree': 1},
                    # {'kernel': 'rbf', 'C': 10, 'degree': 1},
                    {'kernel': 'rbf', 'C': 100, 'degree': 1},
                    #{'kernel': 'rbf', 'C': 1000, 'degree': 1},
                    #{'kernel': 'rbf', 'C': 5000, 'degree': 1},
        #{'kernel': 'rbf', 'C': 50, 'degree': 1},
        #{'kernel': 'rbf', 'C': .001, 'degree': 1},
        #{'kernel': 'rbf', 'C': .0001, 'degree': 1},

        ]

    table, best_score, best_model, params = cross_validation(x, y, xcols, param_set, num_folds= 3)
    print('retraining model with full dataset and best hyperparams' )
    model = SVC(C=params['C'],kernel=params['kernel'],degree=params['degree'], gamma='auto')
    best_model = model.fit(x,y)

    # your code here
    # iterate over all hyperparameter configurations
    # perform 3 FOLD cross validation
    # print cv scores for every hyperparameter and include in pdf report
    # select best hyperparameter from cv scores, retrain model
    return best_model, best_score, x, table

# predict for data in filename test_csv using trained model
def predict(test_csv, trained_model, x):
    x_test, _ , xcols = load_data(test_csv, encode_cat = True, outlier_removal = False, feature_engineering = True)
    x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test) , columns = xcols)
    _,x_test = x.align(x_test, join='outer', axis=1, fill_value=0)
    xcols = list(x.columns)
    xtestcols = list(x_test.columns)
    x_test.drop([x for x in xtestcols if x not in xcols], axis = 1, inplace = True)
    predictions = trained_model.predict(x_test)
    return predictions

# save predictions on test data in desired format
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')
    print('filed saved')

if __name__ == '__main__':
    predictions_saved = False
    submission = False
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    # fill in train_and_select_model(training_csv) to
    # return a trained model with best hyperparameter from 3-FOLD
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter.
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score, x, _ = train_and_select_model(training_csv, submission)



    print(_)
    print ("The best model was scored %.2f" % cv_score)
    # use trained SVC model to generate predictions
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset

    if submission:
        print('Inference not done using this KFOLD run')
        print('This is just a proof of concept to show that code can run - Trained on a subset of the initial data')
        print('Stored prediction file contains inference when training models on full dataset')

    if not predictions_saved:
        predictions = predict(testing_csv, trained_model, x)
        output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
