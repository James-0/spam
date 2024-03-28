import logging
from msilib.schema import tables
import os
import time
from flask import jsonify
import nltk
import joblib
import pandas as pd
import numpy as np
import matplotlib
from sklearn.tree import DecisionTreeClassifier
import socketio
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
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn import metrics
import seaborn as sns
from PIL import Image
import base64
import io
from flask_socketio import SocketIO
import asyncio


algorithm = ['Multi-NB','SVM','KNN', 'RF', 'AdaBoost']
global vectorizer
A = MultinomialNB(alpha=1.0,fit_prior=True)
B = LinearSVC(random_state=0, tol=1e-5)
C = KNeighborsClassifier(n_neighbors=1)
D = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=None)
base_learner = DecisionTreeClassifier(max_depth=1)
E = AdaBoostClassifier(base_learner, n_estimators=50) 
clf = [A,B,C,D,E]
logging.basicConfig(level=logging.INFO)



def show_data(path):
    global data
    data = pd.read_csv(path)
    return data

def custom_train_test_split(data, test_size=0.30, random_state=None, train_size=0.70):
    
    column = data['Recommended']

    labels =  ['spam messages', 'ham messages']

    # Checking the number of true or ham reviews
    ham_count = column[column == 1].count()

    # Checking the number of False or spam reviews
    spam_count = column[column == 0].count()


    data_type_count = {'type' : 'value', 'Recommended' : ham_count, 'Not-recommended' : spam_count}
    
    # didn't clean data 
    print(type(data['Review Text']))


    # Removing stopwords of English
    stopset = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()

    #Initialising Count Vectorizer
    vectorizer = CountVectorizer(stop_words=stopset,binary=True, analyzer='word', ngram_range=(2, 2))
    X = vectorizer.fit_transform(data['Review Text'].values.astype('U'))


    # Extract target column 'Class'
    y = data['Recommended']

    #Performing test train Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, train_size=train_size)
    
    
    return X_train, X_test, y_train, y_test



def train_classifier(model, X_train, y_train):    
    model.fit(X_train, y_train)

    
# function to predict features 
def predict_labels(clf, features):
    return(clf.predict(features))



def save_model(model, index):
    # '''Saving the model to a file'''
    # if filename is None:
    #         # timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"model_{algorithm[index]}.pkl"
    joblib.dump(model, filename)

    return filename




def get_image(y_test, y_pred, index):
    print("prepping picture....")
    model = get_matrix(y_test, y_pred)
    figure = model.get_figure()    
    image_data = io.BytesIO()
    figure.savefig(image_data, format='png')
    return base64.b64encode(image_data.getvalue())


def get_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    return sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')   





# '''Async Functions'''

async def process_algorithm(algorithm_name, index):
    X_train, X_test, y_train, y_test = custom_train_test_split(data)

    logging.info(f"training model {algorithm[index]}")
    model = train_classifier(clf[index], X_train, y_train)
    logging.info(f"ended training for {algorithm[index]}")

    print(f"predicting labels for {algorithm[index]}")
    y_pred = predict_labels(clf[index],X_test)
    logging.info(f"ended predicting labels for {algorithm[index]}")


    logging.info(f"finding f1 score and accuracy for {algorithm[index]}")
    model = f"{algorithm_name}_model"
    accuracy = round(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None), 2)
    f1_scoree = round(f1_score(y_test, y_pred), 2)
    cf_matrix = confusion_matrix(y_test, y_pred)
    recall = round(recall_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred), 2)
    

    logging.info(f"{save_model(model, index)} is saved")

    get_image(y_test, y_pred, index)
    
    return {'name': algorithm_name, 'model': model, 'accuracy': accuracy, 'f1_scoree': f1_scoree, 'cf_matrix': cf_matrix.tolist(), 'recall': recall, 'precision': precision}


# async def generate_algorithm_results(algorithms, emit_function):
#     global isStreamEnded
#     logging.info("generate_algorithm_results() function is now called")
    

#     for index, algorithm in enumerate(algorithms):
#         result = await process_algorithm(algorithm, index)
#         emit_function('stream_data', result)
#     emit_function('stream_end')

