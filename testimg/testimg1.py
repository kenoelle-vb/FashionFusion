from flask import Flask, render_template, send_file
import os

app = Flask(__name__, template_folder='../testimg')

@app.route('/')
def index():
    return render_template('testimg1.htm')

@app.route('/image')
def get_image():
    return send_file("C:/Users/keno/Downloads/TYS Indonesia Guidebook.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)