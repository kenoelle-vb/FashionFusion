from flask import Flask, render_template, request
from flask_dropzone import Dropzone

app = Flask(__name__, template_folder='../imageembed')

@app.route('/')
def index():
    image_url = "https://i.ibb.co.com/s9T9nZ3/White-Shirt.png"
    return render_template('imageembed.htm', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)