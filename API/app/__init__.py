from flask import Flask
from flask import request
from model.voting_classifier import votingClassifier
app = Flask(__name__)


@app.route('/')
def Home():
    return " this the homw page"

@app.route('/predict',methods = ['post'])
def predict():
    data = request.json
    v = votingClassifier()
    return v.predict(data)

if __name__ == "__main__":
    app.run(debug = True)