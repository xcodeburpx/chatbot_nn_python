from flask import Flask, render_template, request
from flask import jsonify

app = Flask(__name__, static_url_path="/static")

@app.route("/message", methods=['POST'])
def reply():
	return jsonify( {'text' : text} )


@app.route("/")
def index():
	return render_template("index.html")

import gen

text = gen.gen()

if __name__ == '__main__':
	app.run(port=5000)