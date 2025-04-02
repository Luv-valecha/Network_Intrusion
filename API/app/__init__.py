import sys,os,json
import pandas as pd
from flask import Flask
from flask import request
app = Flask(__name__)

# Add the parent directory so that the 'scripts' folder is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.voting_classifier import votingClassifier

# Load the classifier
v = votingClassifier()
v.create_voting_classifier()
v.train()


@app.route('/')
def Home():
    return " this the home route"

@app.route('/predict', methods=['POST'])
def predict():
    print("inside predict")

   # Get JSON data from the request
    data = request.json  
    print("Received data:", data)  # Debugging

    # Convert dictionary of lists into a DataFrame
    data_df = pd.DataFrame.from_dict(data)
    print("Converted DataFrame:\n", data_df)  # Debugging

    # Use the classifier to predict
    predictions = v.predict(data_df)  # Assuming `v` is your trained voting classifier
    #print(predictions)

    # Return predictions as a response
    return {"predictions": predictions.tolist()}  # Convert predictions to a list for JSON serialization

if __name__ == "__main__":
    app.run(debug = True)