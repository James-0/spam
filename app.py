import asyncio
from fileinput import filename
import logging
import os
import flask
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from static.python.server import process_algorithm, show_data, process_file_upload, numerical_transformer_standard, numerical_transformer_minmax

app = Flask(__name__)
app.secret_key = b'3d6f45a5fc12445dbac2f59c3b6c7cb1'
socketio = SocketIO(app)
isStreamEnded = False
logging.basicConfig(level=logging.INFO)

accuracy = [0,0,0,0,0]
f1o_score = [0,0,0,0,0]

per_page = 50
   
result = []
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
    if request.method == 'POST':
        file_path = process_file_upload(app, request)
        return redirect(url_for('.html_table', file_path=file_path))
    
    return render_template("index.html", message="Please Try again")

@app.route('/open_test')
def html_table():
    global data
    data = show_data(file_path)
    page = request.args.get('page', type=int, default=1)
    start = (page - 1) * per_page
    end = start + per_page
    data_slice = data.iloc[start:end].to_dict(orient='records')
    columns = data.columns

    total_rows = len(data)
    total_pages = (total_rows - 1) // per_page + 1

    # # Preprocess data with standard scaler and minmax scaler
    # X_processed_standard, y, preprocessor_standard = preprocess_data(data, numerical_transformer_standard)
    # X_processed_minmax, _, preprocessor_minmax = preprocess_data(data, numerical_transformer_minmax)

    return render_template('classification.html', data=data_slice, columns=columns, page=page, total_pages=total_pages)

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

@app.route('/test')
def test():
    return render_template('test.html')


@socketio.on('generate_data')
def handle_generate_data():
    global isStreamEnded
    algorithms = ['Multi-NB','SVM','KNN','RF', 'AdaBoost']
    if data.empty:
        socketio.emit('log_message', {'message': 'Data is empty'})
        return
    def emit_function(event, data):
        if not isStreamEnded:
            socketio.emit(event, data)
    async def run_async_task():
        await generate_algorithm_results(socketio, algorithms, emit_function)
    asyncio.run(run_async_task())


async def generate_algorithm_results(socketio, algorithms, emit_function):
    global isStreamEnded
    for index, algorithm in enumerate(algorithms):
        socketio.emit('log_message', {'message': f'Generating algorithm results for {algorithms[index]}................'})
        result = await process_algorithm(socketio,algorithm, index)
        emit_function('stream_data', result)

    emit_function('stream_end', {'message': 'stream ended'})
    # isStreamEnded = True
    # handle_stream_end()


@socketio.on('stream_end')
def handle_stream_end():
    global isStreamEnded
    socketio.emit("Data stream completly processed. No more data to be received.")

    isStreamEnded = True
    # socketio.stop()
    logging.info("Connection closed")


# @app.route('/request-for-algorithm', methods=['GET'])
# def request_for_algorithm():
#     index = int(request.args.get('index'))
#     algorithms = ['Multi-NB','SVM','KNN', 'RF', 'AdaBoost']
#     response = asyncio.run(process_algorithm(algorithms[index], index))
#     print(response)
#     return jsonify(response)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         comment = request.form['comment']
#         response = [comment]
#         vect = vectorizer.transform(response).toarray()
#         my_prediction = clf.predict(vect)
#     return render_template('predict_result.html', prediction=my_prediction)


if __name__ == '__main__':
    socketio.run(app, debug=True)
