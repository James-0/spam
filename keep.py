import json
from msilib.schema import tables
from flask import Flask, make_response, render_template, render_template_string, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,accuracy_score
from sklearn import metrics



def mainn_function(file):
# reading the data in the csv file
    data = pd.read_csv(file)
    
    data['numClass'] = data['Class'].map({True :1, False :0})
    data['Count']=0
    for i in np.arange(0,len(data.Text)):
        data.loc[i,'Count'] = len(data.loc[i,'Text'])

    data.drop('new_column', inplace=True, axis='columns')
        
    return data

def ffunction(data):

    labels =  ['spam messages', 'ham messages']
    # Checking the number of true or ham reviews
    no_of_true = data[data['numClass'] == 1]
    ham_count  = pd.DataFrame(pd.value_counts(no_of_true['Count'],sort=True).sort_index())

    # Checking the number of False or spam reviews
    no_of_false = data[data['numClass'] == 0]
    spam_count = pd.DataFrame(pd.value_counts(no_of_false['Count'],sort=True).sort_index())
    yes = int(no_of_false['Class'].count())
    no = int(no_of_true['Class'].count())

    # Tasks = [no_of_true['Count'], no_of_false['Count']]
    Tasks = [no, yes]
    my_dict = [{labels:Tasks} for labels, Tasks in zip(labels, Tasks)]
    task = json.dumps(my_dict)
    dataa = {'type' : 'value', 'ham' : no, 'spam' : yes}

    # Cleaning the data 

    #Removing stopwords of English
    stopset = set(stopwords.words("english"))


    #Initialising Count Vectorizer
    vectorizer = CountVectorizer(stop_words=stopset,binary=True, analyzer='word', ngram_range=(2, 2))
    
    X = vectorizer.fit_transform(data.Text)
    # Extract target column 'Class'
    y = data.numClass


    #Performing test train Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.70, random_state=None)

    objects = ('Multi-NB','SVM','KNN', 'RF', 'AdaBoost')

    def train_classifier(clf, X_train, y_train):    
        clf.fit(X_train, y_train)

    # function to predict features 
    def predict_labels(clf, features):
        return(clf.predict(features))

    # Initialize the five models
    A = MultinomialNB(alpha=1.0,fit_prior=True)
    B=  LinearSVC(random_state=0, tol=1e-5)
    C = KNeighborsClassifier(n_neighbors=1)
    D = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
    E = AdaBoostClassifier(n_estimators=100) 


    clf = [A,B,C,D,E]
    acc_score = [0,0,0,0,0]
    fo1_score = [0,0,0,0,0]



    # Checking classifiers
    for a in range(0,5):
        # print(objects[a])
        train_classifier(clf[3], X_train, y_train)
        y_pred = predict_labels(clf[3],X_test)

        fo1_score[a] = f1_score(y_test, y_pred)
        acc_score[a]=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

        
    return acc_score[3], fo1_score, dataa, vectorizer, clf[3]


def testing(data):

    labels =  ['spam messages', 'ham messages']
    # Checking the number of true or ham reviews
    no_of_true = data[data['numClass'] == 1]
    ham_count  = pd.DataFrame(pd.value_counts(no_of_true['Count'],sort=True).sort_index())

    # Checking the number of False or spam reviews
    no_of_false = data[data['numClass'] == 0]
    spam_count = pd.DataFrame(pd.value_counts(no_of_false['Count'],sort=True).sort_index())
    yes = int(no_of_false['Class'].count())
    no = int(no_of_true['Class'].count())

    # Tasks = [no_of_true['Count'], no_of_false['Count']]
    Tasks = [no, yes]
    my_dict = [{labels:Tasks} for labels, Tasks in zip(labels, Tasks)]
    task = json.dumps(my_dict)
    dataa = {'type' : 'value', 'ham' : no, 'spam' : yes}

    # Cleaning the data 

    #Removing stopwords of English
    stopset = set(stopwords.words("english"))


    #Initialising Count Vectorizer
    vectorizer = CountVectorizer(stop_words=stopset,binary=True, analyzer='word', ngram_range=(2, 2))
    
    X = vectorizer.fit_transform(data.Text)
    # Extract target column 'Class'
    y = data.numClass


    #Performing test train Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.70, random_state=None)

    objects = ('Multi-NB','SVM','KNN', 'RF', 'AdaBoost')

    def train_classifier(clf, X_train, y_train):    
        clf.fit(X_train, y_train)

    # function to predict features 
    def predict_labels(clf, features):
        return(clf.predict(features))

    # Initialize the five models
    A = MultinomialNB(alpha=1.0,fit_prior=True)
    B=  LinearSVC(random_state=0, tol=1e-5, dual=False)
    C = KNeighborsClassifier(n_neighbors=1)
    D = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
    E = AdaBoostClassifier(n_estimators=100) 


    clf = [A,B,C,D,E]
    # Checking classifiers
    for a in range(0,5):
        # print(objects[a])
        model = train_classifier(clf[3], X_train, y_train)
        y_pred = predict_labels(clf[3],X_test)
 
        
    return y_test, y_pred, dataa




    A = MultinomialNB(alpha=1.0,fit_prior=True)
    B=  LinearSVC(random_state=0, tol=1e-5)
    C = KNeighborsClassifier(n_neighbors=1)
    D = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
    E = AdaBoostClassifier(n_estimators=100) 
    clf = [A,B,C,D,E]
    acc_score = [0,0,0,0,0]
    fo1_score = [0,0,0,0,0]
    # Checking classifiers
    for a in range(0,5):
        print("\n For ", objects[a])
        print("training.....")
        model = train_classifier(clf[a], X_train, y_train)
        y_pred = predict_labels(clf[a],X_test)
        print("finding f1 score and accuracy...")
        fo1_score[a] = f1_score(y_test, y_pred)
        acc_score[a] = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        print("Accuracy: " + str(acc_score[a]) + " and F-score: " + str(fo1_score[a]))
        #Get the confusion matrix
        if a == 3:
            print("prepping picture....")
            svm = cf_matrix(y_test, y_pred)
            figure = svm.get_figure()    
            image_data = io.BytesIO()
            figure.savefig(image_data, format='png')
            encoded_img_data = base64.b64encode(image_data.getvalue())

            # save the model to disk 
            # pickle.dump(model, open(filenam, 'wb'))
            joblib.dump(model, filenam)