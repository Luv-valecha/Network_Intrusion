import os
import joblib
import json
import pandas as pd
from sklearn.ensemble import VotingClassifier

class votingClassifier:
    def __init__(self,saved_model_path : str):
        self.saved_model_path = saved_model_path
        self.models = self.load_models(self.saved_model_path)
        self.vclassifier = None

    def load_models(self):
        models = {}
        for file in os.listdir(self.saved_model_path):
            filepath = os.path.join(self.saved_model_path, file)
            if file == "votingClassifier.json":
                continue
            elif file.endswith(".pkl"):
                models[file.split(".")[0]] = joblib.load(filepath)
            elif file.endswith(".json"):
                with open(filepath, "r") as f:
                    models[file.split(".")[0]] = json.load(f)
            return models
    
    def create_voting_classifier(self):
        estimators = [(name, model) for name, model in self.models.items() if hasattr(model, 'fit')]
        self.vclassifier =  VotingClassifier(estimators=estimators,voting="hard")
    
    def train(self,data_path : str):
        if(self.vclassifier == None):
            raise ValueError("model is not initialised run create_voting_classifier()")
        data = pd.read_csv(data_path)

        # Get the labels and drop them
        y_train = data['class']
        X_train = data.drop("class")

        # train the classifier
        self.vclassifier.fit(X_train,y_train)

    def predict(self,x_test : pd.DataFrame ):
        if(self.vclassifier == None):
            raise ValueError("model is not initialised run create_voting_classifier()")
        return self.vclassifier.predict(x_test)

# 


