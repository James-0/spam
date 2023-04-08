from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Flask!"

# @app.route('/')
# def index():
#     return render_template('index.html')

#     @app.route('/')
#     def my_link():
#         print ('Hello')

#         return 'Click'

#     if __name__ == '__main__':
#         app.run(debug=True)

