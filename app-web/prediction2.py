import pandas as pd
from joblib import dump,load
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

##############################
classificationResultsPrediction2 = dict()
accuracyPrediction2 = dict()
sorted_accuracyPrediction1 = dict()
##############################

def dataframeEncoding(df): 
    df['Mjob']=df['Mjob'].map({'at_home':0 ,'services':1, 'teacher':2, 'health':3, 'other':4})
    df['reason']=df['reason'].map({'course':0 ,'home':1, 'reputation':2, 'other':3})
    df['guardian']=df['guardian'].map({'mother':0 ,'father':1, 'other':2})
    df['schoolsup']=df['schoolsup'].map({'no':0, 'yes':1})
    df['paid']=df['paid'].map({'no':0, 'yes':1})
    
######### Prediction 2 ###################
def prediction2SVM():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G3']<10,0,1)
    X = df.drop(["G3","pass","lastName","firstName"], axis=1)
    Y=  df['pass']
    svm2 = load('./Models2/svm.pkl')
    s = svm2.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction2['svm'] = result;
    return s;

def prediction2KNN():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G3']<10,0,1)
    X = df.drop(["G3","pass","lastName","firstName"], axis=1)
    Y=  df['pass']
    knn2 = load('./Models2/knn.pkl')
    s = knn2.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction2['knn'] = result;
    return s;

def prediction2NaiveBayes():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G3']<10,0,1)
    X = df.drop(["G3","pass","lastName","firstName"], axis=1)
    Y=  df['pass']
    nb = load('./Models2/naive_bayes.pkl')
    s = nb.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction2['naiveBayes'] = result;
    return s;

def prediction2RandomForest():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G3']<10,0,1)
    X = df.drop(["G3","pass","lastName","firstName"], axis=1)
    Y=  df['pass']
    forest = load('./Models2/random_forest.pkl')
    s = forest.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction2['randomForest'] = result;
    return s;

def prediction2GradiantBoosting():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G3']<10,0,1)
    X = df.drop(["G3","pass","lastName","firstName"], axis=1)
    Y=  df['pass']
    gradiant = load('./Models2/gradient_boosting.pkl')
    s = gradiant.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction2['gradiantBoosting'] = result;
    return s;

def prediction2DecisionTree():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G3']<10,0,1)
    X = df.drop(["G3","pass","lastName","firstName"], axis=1)
    Y=  df['pass']
    decision = load('./Models2/decision_tree.pkl')
    s = decision.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction2['decisionTree'] = result;
    return s;



def prediction2LogisticRegression():
    df = pd.read_csv('test.csv')
    dataframeEncoding(df)
    df['pass']=np.where(df['G3']<10,0,1)
    X = df.drop(["G3","pass","lastName","firstName"], axis=1)
    Y=  df['pass']
    svm2 = load('./Models2/logistic_reg.pkl')
    s = svm2.predict(X)
    result = accuracy_score(Y, s)
    accuracyPrediction2['logisticRegression'] = result;
    return s;

def prediction2Results():
    classificationResultsPrediction2['svm'] = prediction2SVM()
    classificationResultsPrediction2['knn'] = prediction2KNN()
    classificationResultsPrediction2['logisticRegression'] = prediction2LogisticRegression()
    classificationResultsPrediction2['naiveBayes'] = prediction2NaiveBayes()
    classificationResultsPrediction2['randomForest'] = prediction2RandomForest()
    classificationResultsPrediction2['decisionTree'] = prediction2DecisionTree()
    classificationResultsPrediction2['gradiantBoosting'] = prediction2GradiantBoosting()
    global linearRegression  
    linearRegression = list(prediction2LinearRegression())



def prediction2LinearRegression():
    df = pd.read_csv('test.csv')
    df['Mjob']=df['Mjob'].map({'at_home':0 ,'services':1, 'teacher':2, 'health':3, 'other':4})
    df['reason']=df['reason'].map({'course':0 ,'home':1, 'reputation':2, 'other':3})
    df['guardian']=df['guardian'].map({'mother':0 ,'father':1, 'other':2})
    df['schoolsup']=df['schoolsup'].map({'no':0, 'yes':1})
    df['paid']=df['paid'].map({'no':0, 'yes':1})   
    X = df.drop(["G3","lastName","firstName"], axis=1)
    Y=  df['G3']
    svm2 = load('./Models2/linreg_model2.pkl')
    s = svm2.predict(X)
    result = svm2.score(X, Y)
    global notesG3
    notesG3 = Y.tolist()
    global noms 
    noms = df['lastName'].tolist();
    global prenoms
    prenoms = df['firstName'].tolist();
    return s;





def sortingAccurcy():
    sortedList = sorted(accuracyPrediction2.items(), key=lambda x:x[1],reverse=True)
    print('-------------------')
    print(sortedList)
    global s_accuracyPrediction2
    s_accuracyPrediction2 = dict(sortedList)