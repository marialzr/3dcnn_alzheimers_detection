import os

from flask import Flask, render_template, request
from keras import backend as K

app = Flask(__name__)
from service.commands import Commands

commands = Commands()

#upload images
UPLOAD_FOLDER = '..\\uploads'


@app.route("/")
def init():
    return render_template("predict.html")


@app.route("/upload", methods=["POST", "GET"])
def index():
    file = request.files['inputFile']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    full_path_file = os.path.join(UPLOAD_FOLDER, filename)

    if 'Predict' in request.form.values():
        K.clear_session()
        prediction = commands.predict(full_path_file)
        return render_template('predict.html', result=prediction)

    elif 'Show image' in request.form.values():
        file_name1 = filename.split(".")[0]+'1.png'
        file_name2 = filename.split(".")[0]+'2.png'
        store_path_dir = '../ui/static'
        commands.plot(full_path_file, filename, store_path_dir)
        load_path1 = '../static/'+file_name1
        load_path2 = '../static/'+file_name2
        return render_template("predict.html", img_name2=load_path2, img_name1=load_path1)

    elif 'Train' in request.form.values():
        K.clear_session()
        prediction = commands.predict(full_path_file)
        print("pred:", prediction)
        return render_template('predict.html', result='done')


@app.route("/custom", methods=["POST", "GET"])
def train_custom():
    file = request.files['anotherInputFile']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    if 'Train' in request.form.values():
        K.clear_session()
        result = commands.train()
        return render_template('predict.html', result=result)

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(debug=True)
