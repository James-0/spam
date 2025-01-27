import logging
import os
import time

from flask import flash, redirect
import nltk
import joblib
import pandas as pd
import numpy as np
from fileinput import filename
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from werkzeug.utils import secure_filename
from sklearn.utils import shuffle
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, recall_score, precision_score, mean_absolute_error
import seaborn as sns
import base64
import io



per_page = 10
random_seed = 42

ALLOWED_EXTENSIONS = {'csv', 'pdf'}

global data

algorithm = ['Multi-NB','SVM','KNN', 'RF', 'AdaBoost']
global vectorizer
A = MultinomialNB(alpha=1.0,fit_prior=True)
B = LinearSVC(tol=1e-5, random_state=42)
C = KNeighborsClassifier(n_neighbors=1)
D = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=random_seed)
base_learner = DecisionTreeClassifier(max_depth=1)
E = AdaBoostClassifier(base_learner, n_estimators=50, random_state=42) 
clf = [A,B,C,D,E]
logging.basicConfig(level=logging.INFO)

def emit_log(socketio, message):
    socketio.emit('log_message', message)

def show_data(path):
    global data
    data = pd.read_csv(path)
    # data = shuffle(data, random_state=random_seed)
    return data


def train_classifier(socketio, model, X_train, y_train):    
    start_time = time.time()
    modell = model.fit(X_train, y_train)
    end_time = time.time() 
    elapsed_time = end_time - start_time
    emit_log(socketio, {'message': f"Time taken to train {type(model).__name__}: {elapsed_time} seconds"})
    return elapsed_time, modell
 
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
async def process_algorithm(socketio, algorithm_name, index):
    X_train, X_test, y_train, y_test, preprocessor = custom_train_test_split(data)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', clf[index])])

    emit_log(socketio, {'message': f"training model {type(pipeline).__name__}"})
    elapsed_time, modell = train_classifier(socketio, pipeline, X_train, y_train)
    emit_log(socketio, {'message': f"ended training for {type(pipeline).__name__}"})

    print(f"predicting labels for {type(pipeline).__name__}")
    y_pred = predict_labels(pipeline,X_test)
    # emit_log(socketio, {'message': f"ended predicting labels for {type(pipeline).__name__}"})

    emit_log(socketio, {'message': f"Processing all metrics for {type(pipeline).__name__}"})
    mae_train = mean_absolute_error(y_train, y_train)
    emit_log(socketio, {'message': f"Mean Absolute Error for {type(pipeline).__name__}: {mae_train}"})
    accuracy = round(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None), 2)
    f1_scoree = round(f1_score(y_test, y_pred), 2)
    cf_matrix = confusion_matrix(y_test, y_pred)
    recall = round(recall_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred), 2)
    

    emit_log(socketio, {'message': f"{save_model(modell, index)} is saved"})

    # get_image(y_test, y_pred, index)
    
    return {'name': algorithm_name, 'elapsed_time': round(elapsed_time, 2), 'accuracy': accuracy, 'f1score': f1_scoree, 'cf_matrix': cf_matrix.tolist(), 'recall': recall, 'precision': precision, 'mae_train' : mae_train}

def custom_train_test_split(data, test_size=0.30, random_state=None, train_size=0.70):
    data['Review Text'] = data['Review Text'].fillna('')
    # Removing stopwords of English
    stopset = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()
    
    # Extract target column 'Class'
    y = data['Recommended']
    X = data.drop(columns='Recommended')

    text_column = 'Review Text'
    categorical_columns = ['Division Name', 'Department Name', 'Class Name']
    numerical_columns = ['Clothing ID', 'Age', 'Rating', 'Positive Feedback Count']

    # numerical_transformer_standard = StandardScaler()
    numerical_transformer_minmax = MinMaxScaler()
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    text_transformer = CountVectorizer(stop_words=stopset,binary=True, analyzer='word', ngram_range=(2, 2))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns),
            ('text', text_transformer, text_column)
        ])
    

 
    #Performing test train Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, train_size=train_size)
    
    
    return X_train, X_test, y_train, y_test, preprocessor


def process_file_upload(app, request):
    global filename
    UPLOAD_FOLDER = 'static/uploads/..'

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if 'file' not in request.files:
        flash('No file path')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.info("File path is %s" % file_path)
    return file_path

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
