from numpy.lib.function_base import gradient
import pandas as pd
from joblib import dump,load
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
##################################
classificationResultsPrediction1 = dict()
accuracyPrediction1 = dict()
sorted_accuracyPrediction1 = dict()





###################################
def dataframeEncoding(df): 
    df['Mjob']=df['Mjob'].map({'at_home':0 ,'services':1, 'teacher':2, 'health':3, 'other':4})
    df['reason']=df['reason'].map({'course':0 ,'home':1, 'reputation':2, 'other':3})
    df['guardian']=df['guardian'].map({'mother':0 ,'father':1, 'other':2})
    df['schoolsup']=df['schoolsup'].map({'no':0, 'yes':1})
    df['paid']=df['paid'].map({'no':0, 'yes':1})
    


######### Prediction 1 ###################
def prediction1SVM():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G1']<10,0,1)
    X = df.drop(["G1","G3","pass","lastName","firstName"], axis=1)
    Y=df['pass']
    svm1 = load('./Models1/svm.pkl')
    s = svm1.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction1['svm'] = result
    return s;

def prediction1KNN():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df.drop(["lastName","firstName"],axis=1)
    df['pass']=np.where(df['G1']<10,0,1)
    X = df.drop(["G1","G3","pass","lastName","firstName"], axis=1)
    Y=df['pass']
    knn = load('./Models1/knn.pkl')
    print('--------------------')
    print(df)
    print(X)
    print(Y)
    s = knn.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction1['knn'] = result
    return s;

def prediction1LogisticRegression():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)

    df['pass']=np.where(df['G1']<10,0,1)
    X = df.drop(["G1","G3","pass","lastName","firstName"], axis=1)
    Y=df['pass']
    logi = load('./Models1/logistic_reg.pkl')
    s = logi.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction1['logisticRegression'] = result

    return s;

def prediction1NaiveBayes():
    df = pd.read_csv('test.csv')

    dataframeEncoding(df)
    df['pass']=np.where(df['G1']<10,0,1)
    X = df.drop(["G1","G3","pass","lastName","firstName"], axis=1)
    Y=df['pass']
    logi = load('./Models1/naive_bayes.pkl')
    s = logi.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction1['naiveBayes'] = result

    return s;



def prediction1RandomForest():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G1']<10,0,1)
    X = df.drop(["G1","G3","pass","lastName","firstName"], axis=1)
    Y=df['pass']
    randomForest = load('./Models1/random_forest.pkl')
    s = randomForest.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction1['randomForest'] = result

    return s;

def prediction1GradiantBoosting():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G1']<10,0,1)
    X = df.drop(["G1","G3","pass","lastName","firstName"], axis=1)

    Y=df['pass']
    gradientBoosting = load('./Models1/gradient_boosting.pkl')
    s = gradientBoosting.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction1['gradiantBoosting'] = result
    return s;


def prediction1DecisionTree():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G1']<10,0,1)
    X = df.drop(["G1","G3","pass","lastName","firstName"], axis=1)
    Y=df['pass']
    decisonTree = load('./Models1/decision_tree.pkl')
    s = decisonTree.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction1['decisionTree'] = result
    return s;


def prediction1LinearRegression():
    df = pd.read_csv('test.csv')
    global noms 
    noms = df['lastName'].tolist();
    global prenoms
    prenoms = df['firstName'].tolist();
    df = df.drop(df.columns[3],axis=1)
    dataframeEncoding(df)
    X = df.drop(["G1","G3","lastName","firstName"], axis=1)
    print('-----------------')
    print(X)
    Y=  df['G1']
    linearReg = load('./Models1/linreg_model1.pkl')
    s = linearReg.predict(X)
    global notesG1
    notesG1 = Y.tolist()
    return s;


def prediction1Results():
    classificationResultsPrediction1['knn'] = prediction1KNN()
    classificationResultsPrediction1['svm'] = prediction1SVM()
    classificationResultsPrediction1['logisticRegression'] = prediction1LogisticRegression()
    classificationResultsPrediction1['naiveBayes'] = prediction1NaiveBayes()
    classificationResultsPrediction1['randomForest'] = prediction1RandomForest()
    classificationResultsPrediction1['decisionTree'] = prediction1DecisionTree()
    classificationResultsPrediction1['gradiantBoosting'] = prediction1GradiantBoosting()
    global linearRegression  
    linearRegression = list(prediction1LinearRegression())

def sortingAccurcy():
    sortedList = sorted(accuracyPrediction1.items(), key=lambda x:x[1],reverse=True)
    print('-------------------')
    print(sortedList)
    global s_accuracyPrediction1
    s_accuracyPrediction1 = dict(sortedList)

    

