import asyncio
from fileinput import filename
import logging
import os
import flask
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, stream_with_context, url_for, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from serve import process_algorithm, show_data

app = Flask(__name__)
app.secret_key = b'3d6f45a5fc12445dbac2f59c3b6c7cb1'
socketio = SocketIO(app)
isStreamEnded = False
logging.basicConfig(level=logging.INFO)

accuracy = [0,0,0,0,0]
f1o_score = [0,0,0,0,0]

# Pagination settings
per_page = 10
   
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
            flash('No file path')
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
            logging.info("File path is %s" % file_path)
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

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         comment = request.form['comment']
#         response = [comment]
#         vect = vectorizer.transform(response).toarray()
#         my_prediction = clf.predict(vect)
#     return render_template('predict_result.html', prediction=my_prediction)

@app.route('/test')
def test():
    return render_template('test.html')


@socketio.on('generate_data')
def handle_generate_data():
    global isStreamEnded
    # algorithms = ['Multi-NB','SVM','KNN', 'RF', 'AdaBoost']
    algorithms = ['Multi-NB','SVM']
    logging.info("Generating data on button click")
    if data.empty:
        socketio.emit('error', {'response': "Data is empty"})
        return
    
    # Define an emit function to send data to the client
    def emit_function(event, data):
        if not isStreamEnded:
            logging.info("stream is not ended, so this should only run once, I guess")
            socketio.emit(event, data)

    async def run_async_task():
        logging.info("calling generate_algorithm_results(), this should also run once")
        await generate_algorithm_results(algorithms, emit_function)


    # Run the asynchronous task using asyncio.run
    asyncio.run(run_async_task())


async def generate_algorithm_results(algorithms, emit_function):
    global isStreamEnded
    for index, algorithm in enumerate(algorithms):
        logging.info(f"isStreamEnded is {isStreamEnded} so, generate algorithm results for {algorithms[index]}")
        result = await process_algorithm(algorithm, index)
        emit_function('stream_data', result)

    emit_function('stream_end', {'message': 'stream ended'})
    # isStreamEnded = True
    # handle_stream_end()


@socketio.on('stream_end')
def handle_stream_end():
    global isStreamEnded
    # Perform cleanup or additional actions when the data stream ends
    logging.info("Data stream complete. No more data to be received.")

    isStreamEnded = True

    # You can perform additional actions here, such as stopping the listening process
    # For example, you might want to close the connection or perform other cleanup tasks
    # socketio.stop()
    logging.info("Connection closed")



# @app.route('/request-for-algorithm', methods=['GET'])
# def request_for_algorithm():
#     index = int(request.args.get('index'))
#     algorithms = ['Multi-NB','SVM','KNN', 'RF', 'AdaBoost']
#     response = asyncio.run(process_algorithm(algorithms[index], index))
#     print(response)
#     return jsonify(response)



if __name__ == '__main__':
    socketio.run(app, debug=True)
    # # Start the SocketIO server in a separate thread
    # socketio.start_background_task(socketio.run, app, debug=True)

    # # Run the Flask app for non-async routes
    # app.run(debug=True)
