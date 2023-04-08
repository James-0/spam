from fileinput import filename
import os
import pickle
from flask import Flask, flash, redirect, render_template, request, url_for
import flask
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename
from server import ffunction, filenam, test, show_data

app = Flask(__name__)
app.secret_key = b'3d6f45a5fc12445dbac2f59c3b6c7cb1'

accuracy = [0,0,0,0,0]
f1o_score = [0,0,0,0,0]


   
# calling functions
result = []
# global file_path, file
ALLOWED_EXTENSIONS = {'csv', 'pdf'}

UPLOAD_FOLDER = 'static/uploads/..'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global data

@app.route('/index.html')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/index.html', methods=['GET', 'POST'])
def upload_file():
    global file_path
    global filename
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # print("File path is %s" % file_path)
        return redirect(url_for('.html_table', file_path=file_path))
    return render_template("index.html", message="Please Try again")

@app.route('/open_test')
def html_table():
    global data
    data = show_data(file_path)
    return render_template('classification.html', tables =[data.to_html(classes='data', header="true", index=False, justify='center', col_space=2)], titles='', name=filename)

@app.route('/result')
def google_pie_chart():
    global vectorizer
    global clf
    if data.empty:
        print("data is empty")
    result = ffunction(data)
    accuracy, f_score, dataa, vectorizer, clf, img = result
    return render_template('result.html', data=dataa, accuracy=round(accuracy*100, 2), f1='{:.2%}'.format(f_score), img_data = img)
    

@app.route('/login', methods = ['POST'])  
def login():  
    if request.method == "POST":  
        email = request.form['user']  
        password = request.form['password']  
      
    if (password=="admin" and email=="admin@admin.com"):  
        return flask.redirect("/index.html")
    else:  
        return flask.render_template('login.html', messagess="Username or password is incorrect")


@app.route('/')
@app.route('/logout')
def logout():
    return render_template('login.html')

@app.route('/predict')
def open_predict():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        response = [comment]
        vect = vectorizer.transform(response).toarray()
        my_prediction = clf.predict(vect)
    return render_template('predict_result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug = True)  
