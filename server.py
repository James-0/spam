from msilib.schema import tables
import nltk
import joblib
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
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix 
from sklearn import metrics
import seaborn as sns
import logging
from PIL import Image
import base64
import io

filenam = 'finalized_model.pkl'


def show_data(path):
    global data
    data = pd.read_csv(path)
    return data
    

def ffunction(data):
    # print("Recommended Data is:\n", data['Recommended'])

    column = data['Recommended']

    labels =  ['spam messages', 'ham messages']
    # Checking the number of true or ham reviews
    ham_count = column[column == 1].count()

    # Checking the number of False or spam reviews
    spam_count = column[column == 0].count()


    dataa = {'type' : 'value', 'Recommended' : ham_count, 'Not-recommended' : spam_count}
    

    # Cleaning the data 
    # Removing stopwords of English
    stopset = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()

    #Initialising Count Vectorizer
    vectorizer = CountVectorizer(stop_words=stopset,binary=True, analyzer='word', ngram_range=(2, 2))


    X = vectorizer.fit_transform(data['Review Text'].values.astype('U'))
    # Extract target column 'Class'
    y = data['Recommended']


    #Performing test train Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.70, random_state=None)

    objects = ('RF','SVM','KNN', 'Multi-NB', 'AdaBoost')
    # Initialize the five models
    A = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=50, oob_score = True, n_jobs = -1)
    # B=  LinearSVC(random_state=0, tol=1e-5)
    # C = KNeighborsClassifier(n_neighbors=1)
    # D = MultinomialNB(alpha=1.0,fit_prior=True)
    # E = AdaBoostClassifier(n_estimators=30) 
    # clf = [A,B,C,D,E]
    # acc_score = [0,0,0,0,0]
    # fo1_score = [0,0,0,0,0]
    # # Checking classifiers
    # for a in range(0,5):
    # print("\n For ", objects[a])
    print("training.....")
    model = train_classifier(A, X_train, y_train)
    y_pred = predict_labels(A,X_test)
    print("finding f1 score and accuracy...")
    fo1_score = f1_score(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    print("Accuracy: " + str(acc_score) + " and F-score: " + str(fo1_score))
    #Get the confusion matrix
    # if a == 0:
    print("prepping picture....")
    svm = cf_matrix(y_test, y_pred)
    figure = svm.get_figure()    
    image_data = io.BytesIO()
    figure.savefig(image_data, format='png')
    encoded_img_data = base64.b64encode(image_data.getvalue())

    # save the model to disk 
    joblib.dump(model, filenam)
    return acc_score, fo1_score, dataa, vectorizer, A, encoded_img_data.decode('utf-8')


def train_classifier(D, X_train, y_train):    
    D.fit(X_train, y_train)

    
# function to predict features 
def predict_labels(clf, features):
    return(clf.predict(features))

def test(dataa):
    loaded_model = joblib.load(filenam)
    result = loaded_model.predict(dataa)
    print(result)

def cf_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    svm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    return svm     