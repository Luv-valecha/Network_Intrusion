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
