#!/Users/mkerrig/anaconda/bin/Python
import pickle
from pprint import pprint
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.learning_curve import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

def plot_precision_recall_curve(y_true,y_scores,title):

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    print recall_score(y_true,y_scores)
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left")
    plt.show()
def decision_func(probabilities,thresh):
    ret_list = []
    for prob in probabilities:
        if prob <= thresh:
            ret_list.append(0)
        else:                   #if greater than threshhold
            ret_list.append(1)
    return ret_list

def try_model(model,title,df_x,df_y, thresh):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    df_x, df_y, test_size=0.3, random_state=0)
    model.fit(X_train,y_train)
    cross_val = cross_val_score(model,df_x,df_y)
    print float(sum(cross_val))/float(len(cross_val))
    y_predict_prob = model.predict_proba(X_test)
    y_predict =  decision_func(y_predict_prob[:,1],0.1)

    #y_predict =  decision_func(y_predict_prob[:,1], thresh)

    y1 = list(y_test)
    y1 = map(int,y1)
    print confusion_matrix(y1,y_predict)
    # print accuracy_score(y1,y_predict)
    # print recall_score(y1,y_predict)
    acc_score = accuracy_score(y1, y_predict)
    print acc_score
    rec_score = recall_score(y1, y_predict)
    prec_score = precision_score(y1, y_predict)
    return acc_score, rec_score, prec_score

    '''y1 = list(y_test)
    y2 = list(y_predict)
    y1 = map(int,y1)
    y2 = map(int,y2)'''
    #plot_precision_recall_curve(y1,y2,title)

def accuracy_score_vs_prob_thresh(model, title, df_x, df_y):
    threshes = np.linspace(.05, .95, 45)
    accuracies = []
    recalls = []
    precision = []
    for thresh in threshes:
        scores = try_model(model,title,df_x,df_y, thresh)
        accuracies.append(scores[0])
        recalls.append(scores[1])
        precision.append(scores[2])

    return threshes, accuracies, recalls, precision


def load_dataframes_x_y(file_path):
    load_dict = {}
    with open(file_path, 'r') as read:
        load_dict = pickle.load(read)
        read.close()
    df = pd.DataFrame(load_dict)
    df_x = df[['age','sex','thalach',
        'exang','years','famhist',
        'thalrest']]
    df_x['years'] = df_x['years'].map(lambda x: 1 if x > 0 else 0)
    df_y = df['num']
    df_y = df_y.replace({'1': '1','2': '1','3': '1','4':'1'})
    return df_x, df_y

def models_test():

    df_x, df_y = load_dataframes_x_y('holdout_copy.pkl', 'r')

    model = GaussianNB()
    title = 'GaussianNaiveBayes Precision Curve'
    try_model(model,title,df_x,df_y,1)

    model = SVC(kernel='linear',C=2,probability=True)
    title = 'SVC Learning Curve'
    try_model(model,title,df_x,df_y,1)

    model = linear_model.LogisticRegressionCV()
    title = 'LogisticRegressionCV Precision Curve'
    try_model(model,title,df_x,df_y,1)

    model = RandomForestClassifier(n_estimators=45)
    title = 'RandomForestClassifier Precision Curve'
    try_model(model,title,df_x,df_y,1)
#######
    '''threshes, accuracies, recalls, precision = \
    accuracy_score_vs_prob_thresh(model, title, df_x, df_y)
    plt.plot(threshes, accuracies, label='acc', color='g')
    plt.plot(threshes, precision, label='prec', color='b')
    plt.plot(threshes, recalls, label='recall', color='r')
    plt.ylabel('percentage')
    plt.xlabel('threshhold')
    plt.legend()
    plt.show()'''
#########
# models_test()
'''
My Notes:
years isn't too bad
Definately thinking bagging is the way to go
Could we define obesity?
'''
