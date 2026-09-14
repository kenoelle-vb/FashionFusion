from flask import Flask, render_template

app = Flask(__name__, template_folder='../Page 2')

@app.route("/")
def index():
    return render_template("page2.htm")

@app.route("/rec1")
def rec1():
    return render_template("rec1.html")

@app.route("/rec2")
def rec2():
    return "This is rec2 page"

@app.route("/rec3")
def rec3():
    return "This is rec3 page"

@app.route("/bear")
def bear():
    return render_template("bear.htm")

if __name__ == "__main__":
    app.run(debug=True)