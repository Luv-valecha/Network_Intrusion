import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle, json

# Add the parent directory so that the 'scripts' folder is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator 

class GradientBoostingModel:
    def __init__(self, train_data_path: str, random_state: int = 42):
        #Initialize the classifier with the training dataset path
        self.train_data_path = train_data_path
        self.random_state = random_state
        self.model = None
        self._load_train_data()
    
    def _load_train_data(self):
        #Load and preprocess training data.
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Training dataset not found at {self.train_data_path}")
        
        df = pd.read_csv(self.train_data_path)
        self.y_train = df['class']
        self.X_train = df.drop(columns=['class'])

    def train(self, hyperparams_path=None):
        #Training the gradientboost model
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': self.random_state
        }
        
        if hyperparams_path and os.path.exists(hyperparams_path):
            try:
                with open(hyperparams_path, "r") as f:
                    custom_params = json.load(f)
                    params.update(custom_params)
            except Exception as e:
                print(f"Error loading hyperparameters: {e}")
        
        self.model = GradientBoostingClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def predict(self, X):
        #Make predictions on new data
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before predicting.")
        return self.model.predict(X)
    
    #Save the trained model to a file
    def save_model(self, save_path: str):
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before saving.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure the directory exists
        
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
            print("Model saved")

#Main function execution
if __name__ == "__main__":

    # Set training dataset path
    train_dataset_path = r"API\data\processed\train_data.csv" 
    test_dataset_path = r"API\data\processed\test_data.csv"

    #Initialize classifier with training data
    gb_model = GradientBoostingModel(train_data_path=train_dataset_path)
    
    #loading the hyperparameters and training the model
    hyperparams_path = r"API\model\Hyperparams\GradientBoosting_hparam.json"
    gb_model.train(hyperparams_path=hyperparams_path)

    #Save the trained model
    model_save_path = r"API\model\saved_models\gradient_boosting_model.pkl"
    gb_model.save_model(model_save_path)
    
    #evaluate 
    evaluator = ModelEvaluator("gradient_boosting_model.pkl")
    evaluator.evaluate()