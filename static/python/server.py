import logging
import os
import time
from flask import flash, redirect
import nltk
import joblib
import pandas as pd
import numpy as np
from fileinput import filename
from sklearn.calibration import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle
from werkzeug.utils import secure_filename
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, recall_score, precision_score, mean_absolute_error
import seaborn as sns
import base64
import io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



per_page = 10
random_seed = 42

ALLOWED_EXTENSIONS = {'csv', 'pdf'}

data = None
X_processed_standard = None
X_processed_minmax = None
preprocessor_standard = None
preprocessor_minmax = None

logging.basicConfig(level=logging.INFO)

# Define algorithms
A = MultinomialNB(alpha=1.0, fit_prior=True)
B = LinearSVC(tol=1e-5, random_state=42)
C = KNeighborsClassifier(n_neighbors=1)
D = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=42)
base_learner = DecisionTreeClassifier(max_depth=1)
E = AdaBoostClassifier(base_learner, n_estimators=50, random_state=42)
clf = [A, B, C, D, E]
algorithm = ['Multi-NB', 'SVM', 'KNN', 'RF', 'AdaBoost']


def emit_log(socketio, message):
    socketio.emit('log_message', message)

def show_data(path):
    global data
    data = pd.read_csv(path)
    run_preprocess_train()
    data = shuffle(data, random_state=random_seed)
    return data

# Define stopwords and other constants
nltk.download('stopwords')
ps = PorterStemmer()
# stopset = nltk.corpus.stopwords.words('english')
stopset = set(stopwords.words('english'))
text_column = 'Review Text'

categorical_columns = ['Division Name', 'Department Name', 'Class Name']
# data.dropna(subset=categorical_columns, inplace=True)

numerical_columns = ['Age', 'Rating', 'Positive Feedback Count']
imputer = SimpleImputer(strategy='median')
# data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Define preprocessors for categorical and text data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
# text_transformer = CountVectorizer(stop_words=stopset, binary=True, analyzer='word', ngram_range=(2, 2))
text_transformer = CountVectorizer(binary=True, analyzer='word', ngram_range=(2, 2))

# Define preprocessors for numerical data
numerical_transformer_standard = StandardScaler()
numerical_transformer_minmax = MinMaxScaler()

# Function to preprocess data once
def preprocess_data(data, numerical_transformer):
    global y
    data = shuffle(data, random_state=random_seed)
    data['Review Text'] = data['Review Text'].fillna('')
    data['Processed Review Text'] = data['Review Text'].apply(preprocess_text)


    data['Review Length'] = data['Processed Review Text'].apply(lambda x: len(x.split()))

    data.dropna(subset=categorical_columns, inplace=True)

    data.drop_duplicates(inplace=True)

    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns),
            ('text', text_transformer, 'Processed Review Text')
        ])
    
    y = data['Recommended']
    X = data.drop(columns=['Clothing ID', 'Recommended', 'Review Text'])
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


def run_preprocess_train():
    global preprocessor_standard, preprocessor_minmax, X_test_minmax, X_test_standard, X_train_minmax, X_train_standard, y_train, y_test
    # Preprocess data with standard scaler and minmax scaler
    X_processed_standard, y, preprocessor_standard = preprocess_data(data, numerical_transformer_standard)
    X_processed_minmax, _, preprocessor_minmax = preprocess_data(data, numerical_transformer_minmax)

    # Train-test split
    X_train_standard, X_test_standard, y_train, y_test = train_test_split(X_processed_standard, y, test_size=0.3, random_state=42)
    X_train_minmax, X_test_minmax, _, _ = train_test_split(X_processed_minmax, y, test_size=0.3, random_state=42)


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

def mae(model, X_train, X_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    return mae_train, mae_test

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

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word.isalpha() and word not in stopset]
    return ' '.join(tokens)

# '''Async Functions'''
async def process_algorithm(socketio, algorithm_name, index):

    if algorithm_name == 'Multi-NB':
        X_train_selected, X_test_selected = X_train_minmax, X_test_minmax
    else:
        X_train_selected, X_test_selected = X_train_standard, X_test_standard

    pipeline = Pipeline(steps=[('model', clf[index])])

    emit_log(socketio, {'message': f"training model {type(clf[index]).__name__}"})
    elapsed_time, modell = train_classifier(socketio, pipeline, X_train_selected, y_train)
    emit_log(socketio, {'message': f"ended training for {type(clf[index]).__name__}"})

    print(f"predicting labels for {type(clf[index]).__name__}")
    y_pred = predict_labels(pipeline, X_test_selected)
    # emit_log(socketio, {'message': f"ended predicting labels for {type(clf[index]).__name__}"})

    emit_log(socketio, {'message': f"Processing all metrics for {type(clf[index]).__name__}"})
    mae_train, mae_test = mae(pipeline, X_train_selected, X_test_selected)
    emit_log(socketio, {'message': f"Mean Absolute Error for {type(clf[index]).__name__}: {mae_train}"})
    accuracy = round(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None), 2)
    f1_scoree = round(f1_score(y_test, y_pred), 2)
    cf_matrix = confusion_matrix(y_test, y_pred)
    recall = round(recall_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred), 2)
    

    emit_log(socketio, {'message': f"{save_model(modell, index)} is saved"})

    # get_image(y_test, y_pred, index)
    
    return {'name': algorithm_name, 'elapsed_time': round(elapsed_time, 2), 'accuracy': accuracy, 'f1score': f1_scoree, 'cf_matrix': cf_matrix.tolist(), 'recall': recall, 'precision': precision, 'mae_train' : mae_train}

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
