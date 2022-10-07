import math
import statistics
from collections import Counter
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix, recall_score, \
    accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from skfuzzy import control as ctrl
import skfuzzy as fuzz

# Variable to activate plots
to_plot = False

# Random State of functions
random_state = 1

def test_mlp(xtrain, ytrain, hidden_layers: list, to_balance=False,
             to_crossvalidate=False, to_run=False):
    """
    Creates and Evaluates MLPs with or without balancing the X
    set and with or without crossvalidation
    NOTE: We only want to balance the dataset when we don't use crossvalidate
    :param xtrain: xtrain
    :param ytrain: ytrain
    :param hidden_layers: hidden_layers
    :param to_crossvalidate: Runs CrossValidation if True, otherwise normal split
    :param to_balance: Balances X set if True
    :param to_run: Runs function if True
    :return: recalls of the mlps
    """

    if not to_run:
        return

    xvalidation = None
    yvalidation = None

    if not to_crossvalidate:
        # Validation size becomes 25% because 25% is 20% of 80%
        xtrain, xvalidation, ytrain, yvalidation = train_test_split(
            xtrain, ytrain, test_size=0.25, random_state=random_state, shuffle=to_balance)

        if to_balance:
            # Scatter plot of examples by class label BEFORE BALANCE
            # Uses the Mean of all Requests Colums
            plot_by_class_label(xtrain, ytrain, to_plot)

            xtrain, ytrain = SMOTE(random_state=random_state).fit_resample(xtrain, ytrain)

            # Scatter plot of examples by class label AFTER BALANCE
            # Uses the Mean of all Requests Colums
            plot_by_class_label(xtrain, ytrain, to_plot)

    recall_results = []

    for i in range(len(hidden_layers)):
        classifier = MLPClassifier(hidden_layer_sizes=(hidden_layers[i],),
                                   activation='relu', random_state=random_state,
                                   max_iter=2000)

        if to_crossvalidate:
            # Stractified KFold to keep a crossvalidation split with same number of
            # failures in each one
            # Decided to use 3 kfold (11 failures) as 5 would be too small
            # for a good sample size (7 failures)
            kfold = StratifiedKFold(n_splits=3, shuffle=False)
            cross_validate_recall_results = []
            for train_ix, test_ix in kfold.split(xtrain, ytrain):
                # Select rows
                new_xtrain, xvalidation = xtrain[train_ix], xtrain[test_ix]
                new_ytrain, yvalidation = ytrain[train_ix], ytrain[test_ix]

                # Fit and get metric result
                classifier.fit(new_xtrain, new_ytrain)
                cross_validate_recall_results.append(
                    recall_score(
                        yvalidation, classifier.predict(xvalidation)))

            recall_results.append(
                statistics.mean(cross_validate_recall_results))

        else:
            classifier.fit(xtrain, ytrain)
            recall_results.append(
                recall_score(
                    yvalidation, classifier.predict(xvalidation)))

    return recall_results


def create_df_with_n_requests(dataframe: pd.DataFrame, n: int):
    """
    Create new dataframe with the respective nyumber of previous requests
    :param dataframe: dataframe
    :param n: number of previous requeests to select
    :return: Training and Test sets
    """

    new_dataframe = dataframe[['Load', 'Falha']]

    requests_list = []
    for i in range(1, n + 1):
        requests_list.append('Requests-' + str(i))
        new_dataframe['Requests-' + str(i)] = df['Requests'].shift(periods=i)

    new_dataframe.dropna(inplace=True)

    requests_list.append('Load')

    # Decided to Create the DataSets without Shuffling as the order
    # does not matter anymore ----------------DUVIDA
    # Mlp tem em conta a ordem dos dados???
    xtrain, xtest, ytrain, ytest = \
        dataset_split(new_dataframe, requests_list, 'Falha', to_shuffle=True)

    return xtrain, xtest, ytrain, ytest


def first_expert_experiment(dataframe: pd.DataFrame, n_requests=5, to_run=False):
    """
    Test Best MLP for Request-1 to Requests-n using Cross Validation
    and considering Recall Metric
    :param dataframe: dataframe to test
    :param n_requests: Number of previous Requests to consider
    :param to_run: Runs function if True
    """

    if not to_run:
        return

    for n in range(1, n_requests + 1):
        print("\n### Testing for DataSet with Request-" + str(n) + " ###\n")

        xtrain, _, ytrain, _ = create_df_with_n_requests(dataframe, n)

        # Scaling
        scalled_sets = scale_dataset([xtrain])
        xtrain_scaled = scalled_sets[0]

        print(test_mlp(xtrain_scaled, ytrain,
                       [128, 64, 32, 16, 8, 4, 2], to_balance=True, to_run=to_run))


def second_expert_experiment(dataframe: pd.DataFrame, n_requests=5, to_run=False):
    """
    Test Best MLP for Request-1 to Requests-n, in a single column, using Cross Validation
    and considering Recall Metric
    :param dataframe: dataframe to test
    :param n_requests: Number of previous Requests to consider
    :param to_run: Runs function if True
    """

    if not to_run:
        return

    for n in range(1, n_requests + 1):
        print("\n### Testing for DataSet with Request-" + str(n) + " ###\n")

        xtrain, _, ytrain, _ = create_df_with_n_requests(dataframe, n)

        # Scaling
        scalled_sets = scale_dataset([xtrain])
        xtrain_scaled = scalled_sets[0]

        # Try Several Ways to detect high number of requests before failures
        # modes_to_try = ["Mean", "Mean_Recent", "Mean_Oldest", "Max", "Sum"]
        modes_to_try = ["Mean_Oldest"]

        for selected_mode in modes_to_try:
            xtrain_combined = combine_requests(xtrain_scaled, mode=selected_mode)

            # We already got really good results with 4 nodes in the hidden layer but
            # still we will use more to get more information on the best way to detect an
            # high number of requests before failures
            print(test_mlp(xtrain_combined, ytrain,
                           [16, 8, 4, 2], to_balance=True, to_run=to_run))


def dataset_split(dataframe: pd.DataFrame, x_columns: list, y_column: str,
                  test_size_split=0.2, to_shuffle=False):
    """
    Splits DataSet into two sets
    :param dataframe: dataframe
    :param x_columns: columns to give to X
    :param y_column: name of the y column
    :param test_size_split: Split percentage of Test
    :param to_shuffle: shuffle DataSets if True
    :return: Training, Validation and Test sets
    """

    # Train 80% / Test 20%
    xtrain, xtest, ytrain, ytest = train_test_split(dataframe[x_columns],
                                                    dataframe[y_column],
                                                    test_size=test_size_split,
                                                    random_state=random_state,
                                                    shuffle=to_shuffle)

    return xtrain, xtest, ytrain, ytest


def scale_dataset(datasets_to_scale: list):
    """
    Scales Dataset
    :param datasets_to_scale: all datasets to scale
    :return: scalled datasets
    """

    standard_scaler = StandardScaler()

    scalled = []
    for dataset in datasets_to_scale:
        scalled.append(standard_scaler.fit_transform(dataset))

    return scalled


def plotting_confusion_matrix(classifier, xtest, ytest, display_labels, plotting: bool,
                              title: str = "Confusion Matrix of First MLP"):
    """
    Plot confusion matrix with desired configurations
    :param title:
    :param classifier: classifier
    :param xtest: xtest
    :param ytest: ytest
    :param display_labels: display_labels
    :param plotting: Runs function if True
    """

    if not plotting:
        return

    figure = plot_confusion_matrix(classifier, xtest, ytest,
                                   display_labels=display_labels)

    figure.figure_.suptitle(title)
    plt.show()


def plot_by_class_label(xtrain, ytrain, plotting: bool):
    """
    Plots dataset by class with Requests Columns average
    :param xtrain: xtrain
    :param ytrain: ytrain
    :param plotting: Runs function if True
    """

    if not plotting:
        return

    new_xtrain = combine_requests(xtrain)

    counter = Counter(ytrain)
    # print(counter)

    plt.figure()
    for label, _ in counter.items():
        row_ix = np.where(ytrain == label)[0]
        plt.scatter(new_xtrain[row_ix, 0], new_xtrain[row_ix, 1], label=str(label))
    plt.title("Requests-" + str(xtrain.shape[1] - 1))
    plt.legend()
    plt.show()


def combine_requests(xtrain, mode: str = "Mean"):
    export_xtrain_requests = np.ndarray(shape=(xtrain.shape[0], xtrain.shape[1] - 1))
    export_xtrain_requests[:, 0] = xtrain[:, 0]
    # Load is on column 1
    for i in range(2, xtrain.shape[1]):
        export_xtrain_requests[:, i - 1] = xtrain[:, i]

    if mode == "Mean":
        # Mean of all Request columns
        return np.vstack((export_xtrain_requests.mean(axis=1), xtrain[:, 1])).T

    elif mode == "Mean_Recent":
        # Mean where most Recent requests matters the most
        weights = []
        for k in range(xtrain.shape[1]-1):
            weights.append(math.log(k+2))

        weights.reverse()

        return np.vstack((np.average(export_xtrain_requests, axis=1,
                                     weights=weights), xtrain[:, 1])).T

    elif mode == "Mean_Oldest":
        # Mean where Oldest requests matters the most
        weights = []
        for k in range(xtrain.shape[1]-1):
            weights.append(math.log(k+2))

        return np.vstack((np.average(export_xtrain_requests, axis=1,
                                     weights=weights), xtrain[:, 1])).T

    elif mode == "Max":
        # Max of all Request columns
        return np.vstack((np.amax(export_xtrain_requests, axis=1), xtrain[:, 1])).T

    elif mode == "Sum":
        # Sum of all Request columns
        return np.vstack((np.sum(export_xtrain_requests, axis=1), xtrain[:, 1])).T

    else:
        print("--- WRONG INSERTED MODE ---")
        return


#######################################################################################
# 3- Will it Crash or not?

# Import Dataset
df = pd.read_csv('ACI21-22_Proj1IoTGatewayCrashDataset.csv', decimal='.', sep=',')

# Train 80% / Test 20% -> We are goind to use Cross Validation on the training set later
# Decided to Create the DataSets without Shuffling as the order matters
X_train, X_test, y_train, y_test = dataset_split(df, ['Requests', 'Load'], 'Falha')

print((df["Falha"]==0).sum())

# Scaling
scalled_datasets = scale_dataset([X_train, X_test])
X_trainscaled = scalled_datasets[0]
X_testscaled = scalled_datasets[1]

# input layer > hidden layer > output layer
# hidden layer = 2/3 input layer + output layer
# hidden layer < 2x input layer

# Test Best MLP configuration with Cross Validation and Recall as the metric
# We decided to use Recall as our metric because we want to minimize
# False Negatives and get more True Positives
print(test_mlp(X_trainscaled, y_train, [256, 128, 64, 32, 16, 8, 4, 2],
               to_crossvalidate=True, to_run=False))

# Results of testmlp:
# [0.60, 0.63, 0.5126262626262627, 0.40, 0.31, 0.26, 0.0, 0.0]
# Results were pretty low and bad
# Using the rules of thumb for the number of nodes in the hidden layer
# And considering Input Layer size = 2, both 2 and 4 nodes in the hidden layer
# Did not manage to get any True Positives

# Using 128 as our number of nodes as it got the best Recall between the ones we tried
# Classifier might become overfitted as the number of nodes is too high considering the
# Input and output layer sizes

# ----------------------------------------------- Duvida, porquê usar relu e não outro?
# ----------------------------------------------- Duvida, porquê usar número par de nodes?
clf1 = MLPClassifier(hidden_layer_sizes=(128,), activation='relu',
                     random_state=random_state, max_iter=2000)

clf1.fit(X_trainscaled, y_train)
y_pred = clf1.predict(X_testscaled)

print("Recall Score of first MLP: " + str(recall_score(y_test, y_pred)))
print("Accuracy Score of first MLP: " + str(accuracy_score(y_test, y_pred)))
print("Precision Score of first MLP: " + str(precision_score(y_test, y_pred)))
print("F1-Score Score of first MLP: " + str(f1_score(y_test, y_pred)))

# Results:
# Recall Score of first MLP: 0.82
# Accuracy Score of first MLP: 0.98
# Precision Score of first MLP: 0.6
# F1-Score Score of first MLP: 0.69

# The Resulting Recall Score was 0.82 for the test set, much higher than the average one
# with the crossvalidation (0.63). This result is good but we can do better, and the
# difference from the validation to the test set might indicate that this is not
# a good model or that the dataset was not well processed
# Also a low Precision of 0.6 might indicate that the classifier tries to classifie
# some inputs as a failure more times to prevent being wrong and dispite this not being
# bad, its not ideal
# Also a low f1-score of 0.69 might also tell us that we need a better model
# It might be overffited as the number of nodes is too high considering the
# Input and output layer sizes

# Plot Confusion Matrix
plotting_confusion_matrix(clf1, X_testscaled, y_test, ["Non-Crash", "Crash"], to_plot,
                          title="Confusion Matrix of First MLP", )


#######################################################################################
# 4 – Enter the expert…

# Test Best MLP for Request-1 to Requests-n using Cross Validation
# and considering Recall Metric
first_expert_experiment(df, to_run=False)

# Results:
# ### Testing for DataSet with Request-1 ###
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# ### Testing for DataSet with Request-2 ###
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# ### Testing for DataSet with Request-3 ###
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# ### Testing for DataSet with Request-4 ###
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# ### Testing for DataSet with Request-5 ###
# [0.9, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9]

# With these results we cannot be sure what number of requests we should use,
# In this situation we could use other metrics like F1 Score to verify the best choice,
# but fortunetly we with our plots we managed to verify that the best dataframe
# (where the failures are more distanced from non failures and that makes the problem
# almost linearly seperable), is the dataframe where we consider from Requests-1
# to Request-3 so that's the one we are going to use

# As for the number of nodes in the hidden layer we are going to follow the rules of thumb
# and use 4 hidden layer nodes

# We won't tune any more hyperparameters as the model got really good results

# Decided to Create the datasets with shuffling as in this case we have all the
# information we need in one row and the order does not matter
X_train, X_test, y_train, y_test = create_df_with_n_requests(df, 3)

# Scaling
scalled_datasets = scale_dataset([X_train, X_test])
X_trainscaled = scalled_datasets[0]
X_testscaled = scalled_datasets[1]

# Scatter plot of examples by class label BEFORE BALANCE
# Uses the Mean of all Requests Colums
plot_by_class_label(X_trainscaled, y_train, to_plot)

# Balance Dataset
X_balanced_train, y_balanced_train = SMOTE(random_state=random_state).fit_resample(
    X_trainscaled, y_train)

# Scatter plot of examples by class label AFTER BALANCE
# Uses the Mean of all Requests Colums
plot_by_class_label(X_balanced_train, y_balanced_train, to_plot)

# Creating classifier with 4 hidden layer nodes and balanced dataset
clf2 = MLPClassifier(hidden_layer_sizes=(4,), activation='relu',
                     random_state=random_state, max_iter=2000)

clf2.fit(X_balanced_train, y_balanced_train)
y_pred = clf2.predict(X_testscaled)

print("\nRecall Score of second MLP: " + str(recall_score(y_test, y_pred)))
print("Accuracy Score of second MLP: " + str(accuracy_score(y_test, y_pred)))
print("Precision Score of second MLP: " + str(precision_score(y_test, y_pred)))
print("F1-Score Score of second MLP: " + str(f1_score(y_test, y_pred)))

# Results
# Recall Score of second MLP: 1.0
# Accuracy Score of second MLP: 1.0
# Precision Score of second MLP: 1.0
# F1-Score Score of second MLP: 1.0

# All metrics have a score of 1 wich is the best possible outcome
# A different test set, and maybe a bigger one, might help to verify
# if this model is really this good

# Plot Confusion Matrix
plotting_confusion_matrix(clf2, X_testscaled, y_test, ["0", "1"], to_plot)

# We got a perfect Confusion Matrix with this setup


#######################################################################################
# 5 – Let’s get another expert…

# For the seconnd experiment we want to find an abnormal number of requests that will
# probably lead to a Failure
# We will try someways to verify this, like using weighted averages, sums and max values

second_expert_experiment(df, to_run=False)

# By using the Plots we firstly verifies that 3 requests is still the best number of
# requests to predict failure, also we got the following results for each mode of putting
# together the columns:
# Max -> does not separate the data at all, not the way to go
# Sum -> good separation, much like the mean
# Mean -> good separation, as we did already notice on the first experiment
# Mean_Recent -> Worse results that mean considering that the
#               weights follow a log function with reversed values
# Mean_Oldest -> Better results than mean considering that the
#               weights folllow a log function

# With these results we decided to use the Mean_Oldest where oldest requests have a
# litle more weight than recent ones. Also we will keep the 3 previous request values

# Also, here are the results while using Mean_Oldest
# ### Testing for DataSet with Request-1 ###
# [1.0, 1.0, 1.0, 1.0]
# ### Testing for DataSet with Request-2 ###
# [1.0, 1.0, 1.0, 1.0]
# ### Testing for DataSet with Request-3 ###
# [1.0, 1.0, 1.0, 1.0]
# ### Testing for DataSet with Request-4 ###
# [1.0, 1.0, 1.0, 1.0]
# ### Testing for DataSet with Request-5 ###
# [0.9, 0.9, 0.9, 1.0]

# We got very good results for the Validation Set, much like the first experiment
# and with 3 requests especially, we got all recall_score equal to 1

X_train, X_test, y_train, y_test = create_df_with_n_requests(df, 3)

# Scaling
scalled_datasets = scale_dataset([X_train, X_test])
X_trainscaled = scalled_datasets[0]
X_testscaled = scalled_datasets[1]

# Combine all Requests into single column
X_trainscaled = combine_requests(X_trainscaled, mode="Mean_Oldest")
X_testscaled = combine_requests(X_testscaled, mode="Mean_Oldest")

# Scatter plot of examples by class label BEFORE BALANCE
# Uses the Mean of all Requests Colums
plot_by_class_label(X_trainscaled, y_train, to_plot)

# Balance Dataset
X_balanced_train, y_balanced_train = SMOTE(
    random_state=random_state).fit_resample(X_trainscaled, y_train)

# Scatter plot of examples by class label AFTER BALANCE
# Uses the Mean of all Requests Colums
plot_by_class_label(X_balanced_train, y_balanced_train, to_plot)

# Creating classifier with 4 hidden layer nodes and balanced dataset
clf3 = MLPClassifier(hidden_layer_sizes=(4,), activation='relu',
                     random_state=random_state, max_iter=2000)

clf3.fit(X_balanced_train, y_balanced_train)
y_pred = clf3.predict(X_testscaled)

print("\nRecall Score of third MLP: " + str(recall_score(y_test, y_pred)))
print("Accuracy Score of third MLP: " + str(accuracy_score(y_test, y_pred)))
print("Precision Score of third MLP: " + str(precision_score(y_test, y_pred)))
print("F1-Score Score of third MLP: " + str(f1_score(y_test, y_pred)))

# Results
# Recall Score of second MLP: 1.0
# Accuracy Score of second MLP: 0.9925
# Precision Score of second MLP: 0.8
# F1-Score Score of second MLP: 0.89

# These results despite not reaching the same capacity of the MLP of the first expert
# Still got good results and the best possible Recall that is the most important metric
# For this problem

# Plot Confusion Matrix
plotting_confusion_matrix(clf3, X_testscaled, y_test, ["0", "1"], to_plot)

#######################################################################################
# 7 – Fuzzy Rule Based Expert System

# We are going to use the data we obtained on the step 5 of our project

X_train, X_test, y_train, y_test = create_df_with_n_requests(df, 3)

# Scaling
scalled_datasets = scale_dataset([X_train, X_test])
X_trainscaled = scalled_datasets[0]
X_testscaled = scalled_datasets[1]

# Combine all Requests into single column
X_trainscaled = combine_requests(X_trainscaled, mode="Mean_Oldest")
X_testscaled = combine_requests(X_testscaled, mode="Mean_Oldest")

# Balance Dataset
#X_balanced_train, y_balanced_train = SMOTE(
    #random_state=random_state).fit_resample(X_trainscaled, y_train)


all_dataset = np.vstack((X_balanced_train.T, y_balanced_train)).T
test = np.where(all_dataset[:, 2]==1)
only_failures = all_dataset[np.where(all_dataset[:, 2]==1)]
sorted_failures_by_request = only_failures[only_failures[:, 0].argsort()]
sorted_failures_by_load = only_failures[only_failures[:, 1].argsort()]

# Antecedent/Consequent Objects hold universe
# variables and membership functions
min_requests = np.amin(X_balanced_train[:, 0])
max_requests = np.amax(X_balanced_train[:, 0])+0.01

min_load = np.amin(X_balanced_train[:, 1])
max_load = np.amax(X_balanced_train[:, 1])+0.01

requests = ctrl.Antecedent(np.arange(np.amin(X_balanced_train[:, 0]),
                                np.amax(X_balanced_train[:, 0]+0.01), 0.01), 'Average Requests')
load = ctrl.Antecedent(np.arange(np.amin(X_balanced_train[:, 1]),
                                np.amax(X_balanced_train[:, 1]+0.01), 0.01), 'Load')
fail = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'fail')

# Membership function
requests['Low'] = fuzz.trimf(requests.universe, [min_requests, min_requests, max_requests - (max_requests-min_requests)/2])
requests['Normal'] = fuzz.trimf(requests.universe, [max_requests - 3*(max_requests-min_requests)/4, 
                max_requests - (max_requests-min_requests)/2, max_requests - (max_requests-min_requests)/4])
requests['High'] = fuzz.trimf(requests.universe, [max_requests - (max_requests-min_requests)/2, max_requests, max_requests])

load['Low'] = fuzz.trimf(load.universe, [min_load, min_load,  max_load - (max_load-min_load)/2])
load['Normal'] = fuzz.trimf(load.universe, [max_load - 3*(max_load-min_load)/4, 
                max_load - (max_load-min_load)/2, max_load - (max_load-min_load)/4])
load['High'] = fuzz.trimf(load.universe, [max_load - (max_load-min_load)/2, max_load, max_load])

fail['Non-Failure'] = fuzz.trimf(fail.universe, [0, 0, 0.6])
fail['Failure'] = fuzz.trimf(fail.universe, [0.4, 1, 1])

# Plots
requests.view()
load.view()
fail.view()

rule1 = ctrl.Rule(load['High'] & requests['High'], fail['Failure'])
rule2 = ctrl.Rule(load['High'] & requests['Normal'], fail['Failure'])
rule3 = ctrl.Rule(load['High'] & requests['Low'], fail['Non-Failure'])
rule4 = ctrl.Rule(load['Normal'] & requests['High'], fail['Failure'])
rule5 = ctrl.Rule(load['Normal'] & requests['Normal'], fail['Non-Failure'])
rule6 = ctrl.Rule(load['Normal'] & requests['Low'], fail['Non-Failure'])
rule7 = ctrl.Rule(load['Low'] & requests['High'], fail['Failure'])
rule8 = ctrl.Rule(load['Low'] & requests['Normal'], fail['Non-Failure'])
rule9 = ctrl.Rule(load['Low'] & requests['Low'], fail['Non-Failure'])

failCtrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5,
                                   rule6, rule7, rule8, rule9])

# Create a control system simulation
failOutput = ctrl.ControlSystemSimulation(failCtrl)

# Give inputs
failOutput.input['Average Requests'] = X_balanced_train[:, 0]
failOutput.input['Load'] = X_balanced_train[:, 1]

# Crunch the numbers
failOutput.compute()

result = np.round(failOutput.output['fail'])

print("\n--- FUZZY SYSTEM ---")
print("\nRecall Score of fuzzy system: " + str(recall_score(y_balanced_train, result)))
print("Accuracy Score of fuzzy system: " + str(accuracy_score(y_balanced_train, result)))
print("Precision Score of fuzzy system: " + str(precision_score(y_balanced_train, result)))
print("F1-Score Score of fuzzy system: " + str(f1_score(y_balanced_train, result)))

#######################################################################################
# 8 – Generalization

df_test = pd.read_csv('ACI_Proj1_TestSet.csv', decimal='.', sep=',')

# Prepare datasets for created classifiers
new_df_test = df_test[['Load', 'Falha']]

for j in range(1, 4):
    new_df_test['Requests-' + str(j)] = df_test['Requests'].shift(periods=j)

new_df_test.dropna(inplace=True)

X_test, y_test = df_test[['Requests', 'Load']], df_test['Falha']
X_test2, y_test2 = new_df_test[['Requests-1', 'Requests-2', 'Requests-3', 'Load']], \
                   new_df_test['Falha']

# Scaling
scalled_datasets = scale_dataset([X_test, X_test2])
X_testscaled1 = scalled_datasets[0]
X_testscaled2 = scalled_datasets[1]

# Build final set for classifier 3 where Requests are combined
X_testscaled3 = combine_requests(X_testscaled2)
y_test3 = y_test2

# Predict with Classifiers
y_pred1 = clf1.predict(X_testscaled1)
y_pred2 = clf2.predict(X_testscaled2)
y_pred3 = clf3.predict(X_testscaled3)

# Metrics
print("\n--- TEST SET ---")

print("\nRecall Score of first MLP: " + str(recall_score(y_test, y_pred1)))
print("Accuracy Score of first MLP: " + str(accuracy_score(y_test, y_pred1)))
print("Precision Score of first MLP: " + str(precision_score(y_test, y_pred1)))
print("F1-Score Score of first MLP: " + str(f1_score(y_test, y_pred1)))

print("\nRecall Score of second MLP: " + str(recall_score(y_test2, y_pred2)))
print("Accuracy Score of second MLP: " + str(accuracy_score(y_test2, y_pred2)))
print("Precision Score of second MLP: " + str(precision_score(y_test2, y_pred2)))
print("F1-Score Score of second MLP: " + str(f1_score(y_test2, y_pred2)))

print("\nRecall Score of third MLP: " + str(recall_score(y_test3, y_pred3)))
print("Accuracy Score of third MLP: " + str(accuracy_score(y_test3, y_pred3)))
print("Precision Score of third MLP: " + str(precision_score(y_test3, y_pred3)))
print("F1-Score Score of third MLP: " + str(f1_score(y_test3, y_pred3)))

# Confusion Matrixes
plotting_confusion_matrix(clf1, X_testscaled1, y_test, ["0", "1"], to_plot)
plotting_confusion_matrix(clf2, X_testscaled2, y_test2, ["0", "1"], to_plot)
plotting_confusion_matrix(clf3, X_testscaled3, y_test3, ["0", "1"], to_plot)

# Results
# --- TEST SET ---

# Recall Score of first MLP: 0.125
# Accuracy Score of first MLP: 0.94
# Precision Score of first MLP: 0.167
# F1-Score Score of first MLP: 0.143

# Recall Score of second MLP: 1.0
# Accuracy Score of second MLP: 1.0
# Precision Score of second MLP: 1.0
# F1-Score Score of second MLP: 1.0

# Recall Score of third MLP: 1
# Accuracy Score of third MLP: 0.99
# Precision Score of third MLP: 0.8
# F1-Score Score of third MLP: 0.89
