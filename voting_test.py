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
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier


BASE_MODEL = ['age','sex','thalach',
        'exang','years','famhist',
        'thalrest']
BASE_MODEL_W_CHOL = ['age','sex','thalach',
        'exang','years','famhist',
        'thalrest', 'chol']
BASE_MODEL_W_BP = ['age','sex','thalach',
        'exang','years','famhist',
        'thalrest', 'trestbpd']
BASE_MODEL_W_BP_CHOL = ['age','sex','thalach',
        'exang','years','famhist',
        'thalrest', 'chol', 'trestbpd']
def load_dataframes_x_y(file_path, x_inputs_list):
    load_dict = {}
    with open(file_path, 'r') as read:
        load_dict = pickle.load(read)
        read.close()
    df = pd.DataFrame(load_dict)
    df_x = df[x_inputs_list].copy()
    df_x['years'] = (df_x['years'].astype(int) > 2).astype(int)
    df_y = df['num']
    df_y = df_y.replace({'1': '1','2': '1','3': '1','4':'1'})
    return df_x, df_y


def decision_func(probabilities,thresh):
    '''
    Function that takes a list of probabilities and returns a 0
    or a 1 based on a threshold of your choosing. For example,
    a given the following list of probabilities [0.4,0.05,0.1,0.67] and a
    threshold of .4 would return [1,0,0,1]

    Ideally the list of probabilities would be provided from
    model.predict_proba(X), with model being the model you are training
    and X being the list of features you are trying to predict target y with.
    '''

    ret_list = []
    for prob in probabilities:
        if prob <= thresh:
            ret_list.append(0)
        else:                   #if greater than threshhold
            ret_list.append(1)
    return ret_list


def tryVoting(num,x_inputs_list, pa_num=50):
    i =0
    df_x, df_y = load_dataframes_x_y('cleaned_copy.pkl', x_inputs_list)

    KNN = KNeighborsClassifier(n_neighbors=5)
    GNB = GaussianNB()
    RandForest = RandomForestClassifier(n_estimators=45)
    supportVector = SVC(probability=True)
    logRegCV = linear_model.LogisticRegression()
    vote = VotingClassifier(estimators=[('GNB',GNB),
    ('SVC',supportVector),('log',logRegCV)],voting ='soft')# ,('log',logRegCV) #,weights=[2,1,1]
    acc_scores = []
    rec_scores = []
    prec_scores = []
    matrix = np.array([[0,0],[0,0]])
    cross_vals = []
    while i <=num:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        df_x, df_y, test_size=0.3)
        model = logRegCV.fit(df_x,df_y)
        global MODEL
        MODEL = model
        y_predict_prob = model.predict_proba(df_x)
        y_predict = decision_func(y_predict_prob[:,1],.115)
        #y_predict = list(model.predict(df_x).astype(int))
        y_true = list(df_y)
        y_true = map(int,df_y)
        matrix+= confusion_matrix(y_true,y_predict)
        acc_score = accuracy_score(y_true, y_predict)
        rec_score = recall_score(y_true, y_predict)
        prec_score = precision_score(y_true, y_predict)
        acc_scores.append(acc_score)
        rec_scores.append(rec_score)
        prec_scores.append(prec_score)
        #print 'acc_score is: ', acc_score
        #print 'recall score:', rec_score
        #print 'precision score:', prec_score
        cross_val = cross_val_score(model,df_x,df_y, cv=10)
        cross_vals.append( float(sum(cross_val))/float(len(cross_val)))
        i+=1
    test_predict_proba = MODEL.predict_proba(df_x.iloc[pa_num, :])
    test_predict = MODEL.predict(df_x.iloc[pa_num, :])
    print matrix / (num)
    print 'acc_score is: ',float(sum(acc_scores))/float(len(acc_scores))
    print 'rec_score is: ',float(sum(rec_scores))/float(len(rec_scores))
    print 'prec_score is: ',float(sum(prec_scores))/float(len(prec_scores))
    print float(sum(cross_vals))/float(len(cross_vals))
    print "++++++ MODEL COEFFICIENTS +++++++\n", model.coef_
    print "++++++ MODEL INTERCEPT ++++++\n", model.intercept_
    print "ONE PATIENT ++++++++++++++++++  \n", df_x.iloc[pa_num, :], df_y.iloc[pa_num]
    print "model_predict_proba = ", test_predict_proba
    print "model_predict = ", test_predict
print "######################  BASE_MODEL ########################### "
tryVoting(1, BASE_MODEL, pa_num = 100)
print "######################  BASE_MODEL_W_BP ########################### "
tryVoting(1, BASE_MODEL_W_BP, pa_num = 100)
print "######################  BASE_MODEL_W_CHOL ########################### "
tryVoting(1, BASE_MODEL_W_CHOL, pa_num = 100)
print "######################  BASE_MODEL_W_BP_CHOL ########################### "
tryVoting(1, BASE_MODEL_W_BP_CHOL, pa_num = 100)
