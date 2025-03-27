import os,sys
import joblib, pickle
import json
import pandas as pd
from sklearn.ensemble import VotingClassifier

# Add the parent directory so that the 'scripts' folder is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator 

class votingClassifier:
    def __init__(self,saved_model_path : str = r"API\model\saved_models"):
        self.saved_model_path = saved_model_path
        self.models = self.load_models()
        self.vclassifier = None

    def load_models(self):
        models = {}
        for file in os.listdir(self.saved_model_path):
            filepath = os.path.join(self.saved_model_path, file)
            if file == "voting_model.pkl":
                continue
            elif file.endswith(".pkl"):
                #print(filepath)
                models[file.split(".")[0]] = joblib.load(filepath)
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
        X_train = data.drop("class",axis=1)

        # train the classifier
        self.vclassifier.fit(X_train,y_train)

    def predict(self,x_test : pd.DataFrame ):
        if(self.vclassifier == None):
            raise ValueError("model is not initialised run create_voting_classifier()")
        return self.vclassifier.predict(x_test)

    def save_model(self, save_path: str = r"API\model\saved_models\voting_model.pkl"):
        """Save the trained model to a file."""
        if self.vclassifier is None:
            raise ValueError("Model is not trained. Call train() before saving.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        with open(save_path,"wb") as f:
            pickle.dump(self.vclassifier,f)
            print(" Model saved")

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"API\data\processed\train_data.csv" 
    test_dataset_path = r"API\data\processed\test_data.csv"
    # Initialize classifier with training data
    classifier = votingClassifier()
    classifier.create_voting_classifier()
    classifier.train(train_dataset_path)

    # Save the trained model
    model_save_path = r"API\model\saved_models\voting_model.pkl"
    classifier.save_model(model_save_path)

    # evaluate
    evaluater = ModelEvaluator("voting_model.pkl")
    evaluater.evaluate()