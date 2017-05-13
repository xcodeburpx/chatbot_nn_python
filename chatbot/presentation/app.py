from flask import Flask, render_template, request
from flask import jsonify
import gen

app = Flask(__name__,static_url_path="/static")

#############
# Routing
#
@app.route('/message', methods=['POST'])
def reply():
    result = gen.gen(request.form['msg'])
    return jsonify( { 'text': result } )

@app.route("/")
def index():
    return render_template("index.html")


# start app
if (__name__ == "__main__"):
    app.run(port = 5000)
