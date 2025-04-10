import os,sys
import joblib, pickle
import json
import pandas as pd
from sklearn.ensemble import VotingClassifier

# Add the parent directory so that the 'scripts' folder is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator 


from BernoulliNB import BernoulliNB
from random_forest_classifier import RandomForestClassifier
from Decision_Tree import DecisionTree, DecisionTreeNode
from knn_classifier import KNNClassifier
from LinearSVC import LinearSVM
from LogisticRegression import LogisticRegression
from Non_LinearSVC import RBFKernelSVM
from ANN import NeuralNetwork

class votingClassifier:
    def __init__(self,saved_model_path : str = r"saved_models"):
        self.saved_model_path = saved_model_path
        self.models = {}
        self.vclassifier = None

    def load_models(self):
        models = {}
        for file in os.listdir(self.saved_model_path):
            filepath = os.path.join(self.saved_model_path, file)
            if file == "voting_model.pkl" or file=="class_encoder.pkl" or file=="protocol_type_encoder.pkl" or file=="flag_encoder.pkl" or file=="service_encoder.pkl":
                continue
            elif file.endswith(".pkl"):
                models[file.split(".")[0]] = joblib.load(filepath)
        return models
    
    def load_pretrained_model(self,model_name : str):
        """
        Load a pre-trained model from the specified path.
        """
        for file in os.listdir(self.saved_model_path):
            filepath = os.path.join(self.saved_model_path, file)
            if file == "voting_model.pkl":
                self.vclassifier = joblib.load(filepath)
                return
        
        raise FileNotFoundError(f"Model {model_name} not found in {self.saved_model_path}")
                

    
    def create_voting_classifier(self):
        self.models = self.load_models()
        estimators = [(name, model) for name, model in self.models.items() if hasattr(model, 'fit')]
        self.vclassifier =  VotingClassifier(estimators=estimators,voting="hard")
    
    def train(self,data_path : str = r"..\data\processed\train_data.csv"):
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

    def save_model(self, save_path: str = r"\saved_models\voting_model.pkl"):
        """Save the trained model to a file."""
        if self.vclassifier is None:
            raise ValueError("Model is not trained. Call train() before saving.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        with open(save_path,"wb") as f:
            pickle.dump(self.vclassifier,f)
            print(" Model saved")

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"..\data\processed\train_data.csv" 
    test_dataset_path = r"..\data\processed\test_data.csv"
    # Initialize classifier with training data
    classifier = votingClassifier()
    classifier.create_voting_classifier()
    classifier.train()

    # Save the trained model
    model_save_path = r"\saved_models\voting_model.pkl"
    classifier.save_model(model_save_path)

    # evaluate
    evaluater = ModelEvaluator("voting_model.pkl")
    evaluater.evaluate()